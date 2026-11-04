#!/usr/bin/env python3
"""stage1_attribution.py —— 记忆介入的「可归因性」体检

Stage 1（离线，不跑仿真、不用 GPU）。只读现有的 episodes.json，回答四个问题：

  Q1 基底　记忆到底介入了多少？（检索命中率、翻转率、邻居数）
  Q2 信号　有没有可用的环境终局结果？（有多少成功回合）
  Q3 集中　影响是不是集中在少数条目上？（这是能不能做归因的生死线）
  Q4 错位　条目上挂的 Qwen 回报，与环境终局成功是否对不上？

不产生任何训练信号，只做描述统计。目的是：在写 learner 之前，
先判断「把终局成败归因到单条记忆」这件事有没有抓手。

用法：
    python3 scripts/stage1_attribution.py                 # 全量
    python3 scripts/stage1_attribution.py --limit 200     # 只扫前 200 个 run
    python3 scripts/stage1_attribution.py --out /tmp/s1   # 指定输出目录
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

GAMMA = 0.95


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def spearman(xs: list[float], ys: list[float]) -> float | None:
    """秩相关。不做并列值修正（并列按平均秩处理）。"""
    n = len(xs)
    if n < 3:
        return None

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def gini(values: list[float]) -> float | None:
    """非负值的基尼系数。0 = 完全摊平，1 = 全部集中在一条。"""
    v = sorted(x for x in values if x > 0)
    n = len(v)
    if n < 2:
        return None
    total = sum(v)
    if total <= 0:
        return None
    cum = 0.0
    for i, x in enumerate(v, start=1):
        cum += i * x
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def collections_bucket(rows: list[dict]) -> list[tuple[str, int]]:
    """每条例目拿到几次「改选命中」事件 —— 决定 v_i 能不能被估出来。"""
    b = {"0（从未命中）": 0, "1": 0, "2–3": 0, "4–9": 0, "≥10": 0}
    for r in rows:
        k = r["n_credit_events"]
        if k == 0:
            b["0（从未命中）"] += 1
        elif k == 1:
            b["1"] += 1
        elif k <= 3:
            b["2–3"] += 1
        elif k <= 9:
            b["4–9"] += 1
        else:
            b["≥10"] += 1
    return list(b.items())


def top_share(values: list[float], frac: float) -> float | None:
    """前 frac 比例的条目占了多少正信用。"""
    v = sorted((x for x in values if x > 0), reverse=True)
    if not v or sum(v) <= 0:
        return None
    k = max(1, int(round(len(v) * frac)))
    return sum(v[:k]) / sum(v)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def iter_runs(root: str, limit: int | None):
    """产出 (run_dir, episodes_path)。run_dir 的父目录名是 method。"""
    paths = sorted(glob.glob(os.path.join(root, "**", "episodes.json"), recursive=True))
    for i, p in enumerate(paths):
        if limit is not None and i >= limit:
            return
        run_dir = os.path.dirname(p)
        method = os.path.basename(os.path.dirname(run_dir))
        yield run_dir, method, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join("artifacts", "stage1_attribution")
    os.makedirs(out_dir, exist_ok=True)

    run_rows: list[dict] = []
    # (run_dir, entry_id) -> 统计
    entry_stats: dict[tuple[str, int], dict] = defaultdict(
        lambda: {
            "n_retrieved": 0,
            "n_flip_when_retrieved": 0,
            "n_retrieved_in_success": 0,
            "credit_env": 0.0,          # 挂上翻转门控 + 终局成败
            "credit_env_noflip": 0.0,   # 不加翻转门控，对照
            "n_credit_events": 0,       # 在「改选」那一步被命中的次数（与成败无关）
            "exposure": 0.0,            # 同一批事件的折扣暴露量，不含 y —— 用来做基线扣减
            "origin_return": None,      # 条目自己那条 Qwen 回报
        }
    )

    n_files = 0
    n_bad = 0
    slot_total = 0          # 邻居槽位总数（对账用）
    key_missing = 0         # 邻居缺 id 计数
    changed_true = 0        # choice_changed 为真的步数
    flipped_key_hit = 0     # 条目 action_key 命中 updated choice 的次数
    run_keys_seen: set[str] = set()
    ids_per_run: dict[str, set[int]] = defaultdict(set)
    method_runs: dict[str, dict] = defaultdict(
        lambda: {"runs": 0, "episodes": 0, "successes": 0, "chunks": 0, "ret": 0, "flip": 0}
    )
    n_chunks_total = 0
    n_chunks_with_neighbors = 0
    n_chunks_flip = 0
    n_chunks_flip_and_ret = 0
    neighbor_counts: list[int] = []
    successes_total = 0
    episodes_total = 0
    runs_with_success = 0
    jitrl_runs = 0

    print(f"扫描 {args.root}/**/episodes.json ...", flush=True)
    for run_dir, method, path in iter_runs(args.root, args.limit):
        n_files += 1
        try:
            with open(path, encoding="utf-8") as fh:
                eps = json.load(fh)
        except Exception:
            n_bad += 1
            continue
        if isinstance(eps, dict):
            eps = eps.get("episodes", [])
        if not isinstance(eps, list) or not eps:
            n_bad += 1
            continue

        # 带记忆的方法：jitrl / jitrl-free / jitrl-free-cycle
        # 对照：static*（advantage 恒 0，不读写 memory）、direct*（不调 Qwen）
        is_memory = method.startswith("jitrl")
        if is_memory:
            jitrl_runs += 1
        method_runs[method]["runs"] += 1
        run_key = os.path.relpath(run_dir, args.root)

        succ = 0
        eps_ok = 0
        run_ret = 0
        run_flip = 0
        run_chunks = 0

        for ep in eps:
            if not isinstance(ep, dict):
                continue
            chunks = ep.get("chunks") or []
            if not chunks:
                continue
            eps_ok += 1
            y = 1 if ep.get("success") else 0
            succ += y
            T = len(chunks)
            for c in chunks:
                if not isinstance(c, dict):
                    continue
                run_chunks += 1
                changed = bool(c.get("choice_changed"))
                if changed:
                    run_flip += 1
                if not is_memory:
                    continue
                n_chunks_total += 1
                if changed:
                    n_chunks_flip += 1
                nbrs = c.get("retrieved_neighbors") or []
                if nbrs:
                    run_ret += 1
                    n_chunks_with_neighbors += 1
                    neighbor_counts.append(len(nbrs))
                if nbrs:
                    if changed:
                        n_chunks_flip_and_ret += 1

                    # 折扣到该步；disc 乘上终局成败，disc_unit 不乘
                    t = int(c.get("chunk_index", 0))
                    disc_unit = GAMMA ** max(0, T - 1 - t)
                    disc = disc_unit * y

                    # 记忆真的改了选择的那一侧（updated choice 落在哪个候选上）
                    cands = c.get("candidates") or []
                    uidx = c.get("updated_choice_index")
                    ukey = None
                    if isinstance(uidx, int) and 0 <= uidx < len(cands):
                        ukey = cands[uidx].get("semantic_key")

                    # 每个 action_key 在本次检索里出现几次 —— 决定 1/n_a 权重
                    per_key: dict[str, int] = defaultdict(int)
                    for nb in nbrs:
                        k = nb.get("action_key")
                        if k is not None:
                            per_key[k] += 1

                    slot_total += len(nbrs)
                    for nb in nbrs:
                        eid = nb.get("id")
                        if eid is None:
                            key_missing += 1
                            continue
                        run_keys_seen.add(run_key)
                        ids_per_run[run_key].add(int(eid))
                        st = entry_stats[(run_key, int(eid))]
                        st["n_retrieved"] += 1
                        if y:
                            st["n_retrieved_in_success"] += 1
                        if changed:
                            st["n_flip_when_retrieved"] += 1
                        if st["origin_return"] is None:
                            st["origin_return"] = float(nb.get("return", 0.0))

                        n_a = per_key.get(nb.get("action_key"), 0)
                        w = (1.0 / n_a) if n_a else 0.0
                        # 只在「这条条目的动作类型正好是被选中的那一个」时计入
                        if ukey is not None and nb.get("action_key") == ukey:
                            flipped_key_hit += 1
                            changed_true += 1 if changed else 0
                            st["credit_env_noflip"] += w * disc
                            if changed:
                                st["credit_env"] += w * disc
                                st["exposure"] += w * disc_unit
                                st["n_credit_events"] += 1

        episodes_total += eps_ok
        successes_total += succ
        if succ > 0:
            runs_with_success += 1
        mr = method_runs[method]
        mr["episodes"] += eps_ok
        mr["successes"] += succ
        mr["chunks"] += run_chunks
        mr["ret"] += run_ret
        mr["flip"] += run_flip
        run_rows.append(
            {
                "run": run_key,
                "method": method,
                "episodes": eps_ok,
                "mean_steps": (
                    statistics.fmean([e.get("steps", 0) for e in eps if isinstance(e, dict)])
                    if eps_ok else 0.0
                ),
                "mean_sum_reward": (
                    statistics.fmean([e.get("sum_reward", 0.0) for e in eps if isinstance(e, dict)])
                    if eps_ok else 0.0
                ),
                "successes": succ,
                "success_rate": (succ / eps_ok) if eps_ok else 0.0,
                "chunks": run_chunks,
                "retrieval_coverage": (run_ret / run_chunks) if run_chunks else 0.0,
                "choice_change_rate": (run_flip / run_chunks) if run_chunks else 0.0,
            }
        )
        if n_files % 200 == 0:
            print(f"  ...{n_files} runs", flush=True)

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Q1 基底：记忆介入了多少")
    print("=" * 72)
    print(f"  扫到 run 目录            {n_files}（读失败 {n_bad}）")
    print(f"  其中 jitrl（带记忆）     {jitrl_runs}")
    print(f"  高层决策步总数           {n_chunks_total}")
    if n_chunks_total:
        print(f"  其中有检索命中           {n_chunks_with_neighbors}"
              f"（{n_chunks_with_neighbors / n_chunks_total:.1%}）")
        print(f"  其中选择被记忆改动       {n_chunks_flip}"
              f"（{n_chunks_flip / n_chunks_total:.1%}）")
        print(f"  改动且当步有检索命中     {n_chunks_flip_and_ret}")
    if neighbor_counts:
        print(f"  邻居数 中位 {statistics.median(neighbor_counts):.0f}"
              f" / 均值 {statistics.fmean(neighbor_counts):.1f}"
              f" / 最大 {max(neighbor_counts)}")

    # ---- 对账：先证明记账没错，再谈结论 ----
    sum_retrieved = sum(s["n_retrieved"] for s in entry_stats.values())
    ids_total = sum(len(v) for v in ids_per_run.values())
    print("\n" + "-" * 72)
    print("对账（数字对不上就不该信后面的结论）")
    print("-" * 72)
    print(f"  邻居槽位总数（预期）     {slot_total}")
    print(f"  条目累计被检索次数（实收）{sum_retrieved}")
    print(f"  差                     {slot_total - sum_retrieved}"
          f"（缺 id 的邻居 {key_missing}）")
    print(f"  不同 run_key             {len(run_keys_seen)}")
    print(f"  各 run 的 id 并集大小    {ids_total}")
    print(f"  条目统计表条目数         {len(entry_stats)}")
    print(f"  命中 updated choice 的槽位 {flipped_key_hit}"
          f"（其中当步 choice_changed 的 {changed_true}）")

    print("\n  分方法：")
    print(f"    {'方法':<20}{'run':>5}{'回合':>7}{'成功':>6}{'成功率':>9}"
          f"{'决策步':>8}{'检索命中':>10}{'改选率':>9}")
    for m in sorted(method_runs, key=lambda k: -method_runs[k]["runs"]):
        s = method_runs[m]
        sr = (s["successes"] / s["episodes"]) if s["episodes"] else 0.0
        cr = (s["chunks"])
        cov = (s["ret"] / cr) if cr else 0.0
        fl = (s["flip"] / cr) if cr else 0.0
        print(f"    {m:<20}{s['runs']:>5}{s['episodes']:>7}{s['successes']:>6}"
              f"{sr:>8.1%}{cr:>8}{cov:>10.1%}{fl:>9.1%}")

    print("\n" + "=" * 72)
    print("Q2 信号：有没有可用的环境终局结果")
    print("=" * 72)
    print(f"  回合总数                 {episodes_total}")
    print(f"  成功回合                 {successes_total}"
          f"（{successes_total / episodes_total:.1%}）" if episodes_total else "")
    print(f"  至少有一个成功的 run     {runs_with_success} / {n_files}")
    if runs_with_success == 0:
        print("  ⚠ 没有任何成功回合 —— 终局成败当监督信号时，全部信用恒为 0")

    # ------------------------------------------------------------------
    rows = []
    for (run_key, eid), st in entry_stats.items():
        rows.append(
            {
                "run": run_key,
                "entry_id": eid,
                "n_retrieved": st["n_retrieved"],
                "n_flip_when_retrieved": st["n_flip_when_retrieved"],
                "n_retrieved_in_success": st["n_retrieved_in_success"],
                "credit_env": st["credit_env"],
                "credit_env_noflip": st["credit_env_noflip"],
                "n_credit_events": st["n_credit_events"],
                "exposure": st["exposure"],
                "origin_return": st["origin_return"] if st["origin_return"] is not None else 0.0,
            }
        )
    # 基线扣减：把「这条条目是在成功回合里出的力多，还是单纯出力的机会多」分开
    p_base = (successes_total / episodes_total) if episodes_total else 0.0
    for r in rows:
        r["credit_signed"] = r["credit_env"] - p_base * r["exposure"]

    # ---- Q2b 配对：记忆到底有没有改变成功率 ----
    print("\n" + "-" * 72)
    print("Q2b 配对比较（同实验、同任务、同 seed，只差方法）")
    print("-" * 72)
    pair: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for r in run_rows:
        parts = r["run"].split(os.sep)
        if len(parts) < 4:
            continue
        exp, task, seed = parts[0], parts[-3], parts[-1]
        pair[(exp, task, seed)][r["method"]] = r

    def paired(a: str, b: str, field: str):
        ds = [d[a][field] - d[b][field] for d in pair.values() if a in d and b in d]
        if len(ds) < 5:
            return None
        mean = statistics.fmean(ds)
        se = statistics.stdev(ds) / math.sqrt(len(ds))
        wins = sum(1 for x in ds if x > 1e-9)
        losses = sum(1 for x in ds if x < -1e-9)
        t = (mean / se) if se > 0 else 0.0
        return len(ds), mean, se, t, wins, losses

    print(f"  {'对比':<40}{'n':>5}{'均值差':>11}{'标准误':>10}{'t':>8}{'胜/负':>10}")
    for a, b in (("jitrl", "static"), ("jitrl-free", "static-free"),
                 ("jitrl-free-cycle", "static-free-cycle")):
        for field, unit, scale in (("success_rate", "成功率", 1.0),
                                   ("mean_steps", "平均步数", 1.0),
                                   ("mean_sum_reward", "平均累计奖励", 1.0)):
            r = paired(a, b, field)
            if r is None:
                continue
            n, mean, se, t, w, l = r
            fmt = f"{mean:+.4f}" if field != "success_rate" else f"{mean:+.2%}"
            sef = f"{se:.4f}" if field != "success_rate" else f"{se:.2%}"
            print(f"  {a+' - '+b+'  ['+unit+']':<40}{n:>5}{fmt:>11}{sef:>10}{t:>+8.2f}{f'{w}/{l}':>10}")
        print()

    # 分任务异质性：效应是不是只在少数任务上
    per_task: dict[str, list[float]] = defaultdict(list)
    for (exp, task, seed), d in pair.items():
        if "jitrl" in d and "static" in d:
            per_task[task].append(d["jitrl"]["success_rate"] - d["static"]["success_rate"])
    if per_task:
        rows_t = sorted(
            ((tk, statistics.fmean(v), len(v)) for tk, v in per_task.items() if len(v) >= 3),
            key=lambda x: -abs(x[1]),
        )
        print(f"  分任务（|差| 最大的 10 个，n≥3 对）：")
        for tk, m, n in rows_t[:10]:
            print(f"    {tk:<34} {m:+.1%}  (n={n})")
        big = sum(1 for _, m, _ in rows_t if abs(m) >= 0.15)
        print(f"    |差| ≥15pt 的任务数：{big} / {len(rows_t)}")

    print("\n" + "=" * 72)
    print("Q3 集中：影响是不是集中在少数条目上")
    print("=" * 72)
    print(f"  被检索过的条目总数       {len(rows)}")
    if rows:
        never_flip = sum(1 for r in rows if r["n_flip_when_retrieved"] == 0)
        print(f"  从未在「改动选择」那一步被检索到的条目  {never_flip}"
              f"（{never_flip / len(rows):.1%}）")
        ret_counts = [r["n_retrieved"] for r in rows]
        print(f"  被检索次数 中位 {statistics.median(ret_counts):.0f}"
              f" / 均值 {statistics.fmean(ret_counts):.1f}"
              f" / 最大 {max(ret_counts)}")

        # 事件数分桶 —— 决定 v_i 能不能被估出来
        ev = collections_bucket(rows)
        print("\n  【每条条目拿到几次「改选命中」事件】")
        for k, v in ev:
            print(f"    {k:<12}{v:>7}（{v / len(rows):.1%}）")

        for label, key in (("带翻转门控 credit_env（恒非负）", "credit_env"),
                           ("基线扣减后 credit_signed（有正有负）", "credit_signed"),
                           ("不带门控 credit_env_noflip", "credit_env_noflip")):
            vals = [abs(r[key]) for r in rows]
            nz = [v for v in vals if v > 0]
            g = gini(nz)
            ts10 = top_share(nz, 0.10)
            print(f"\n  【{label}】非零条目 {len(nz)} / {len(rows)}")
            if g is not None:
                extra = f"（前 10% 占 {ts10:.1%}）" if ts10 is not None else ""
                print(f"    基尼 {g:.3f}{extra}")
            else:
                print("    ⚠ 没有非零信用 —— 这条信号在当前数据上是空的")

    print("\n" + "=" * 72)
    print("Q4 错位：条目上挂的 Qwen 回报 vs 环境终局归因")
    print("=" * 72)
    if rows:
        for label, key in (("credit_env", "credit_env"),
                           ("credit_signed", "credit_signed")):
            pairs = [(r["origin_return"], r[key]) for r in rows if r[key] != 0]
            if len(pairs) >= 3:
                rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
                shown = f"{rho:.3f}" if rho is not None else "无法计算"
                print(f"  {key:<14} 非零条目 {len(pairs):>6}   "
                      f"Spearman(origin_return, {key}) = {shown}")
            else:
                print(f"  {key:<14} ⚠ 非零太少，无法算秩相关")
        print("  解读：接近 0 或为负 ⇒ 条目上挂的 Qwen 回报对「这条到底有没有帮忙」"
              "几乎没有预测力，正是报告里说的那处错位")

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------
    with open(os.path.join(out_dir, "runs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(run_rows[0].keys()) if run_rows else ["run"])
        w.writeheader()
        w.writerows(run_rows)

    if rows:
        with open(os.path.join(out_dir, "entries.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: -abs(r["credit_env"])))

    print(f"\n结果已写出：{out_dir}/runs.csv, entries.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
