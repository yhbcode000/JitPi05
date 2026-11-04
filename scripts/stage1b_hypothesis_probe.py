#!/usr/bin/env python3
"""stage1b_hypothesis_probe.py —— 三个假设的判据探针（离线，零 rollout）

H1 键检索没区分度：状态摘要被模板 token 主导 ⇒ Jaccard 分不开 ⇒ top-10 近随机
H2 VLA 底层执行不了：失败集中在早期执行阶段，换不换高层动作都一样
H3 存进去的分数是噪声：邻居 return 与相似度无关、chunk 分数与回合成败对不上
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")


def toks(s: str) -> set[str]:
    return set(TOKEN_RE.findall((s or "").lower()))


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
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
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def pct(v, q):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or os.path.join("artifacts", "stage1b_probe")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 收集 ----------
    summaries: dict[str, list[str]] = defaultdict(list)      # task -> state_summary
    sims: list[float] = []                                   # 所有邻居的 similarity
    per_chunk_sim: list[tuple[float, float]] = []            # (top1_sim, kth_sim)
    sim_ret: list[tuple[float, float]] = []                  # (similarity, neighbor return)
    steps_ok: list[int] = []
    steps_fail: list[int] = []
    term = Counter()
    chunk_ret_ok: list[float] = []
    chunk_ret_fail: list[float] = []
    n_chunks = 0
    n_files = 0

    for path in sorted(glob.glob(os.path.join(args.root, "**", "episodes.json"), recursive=True)):
        if args.limit is not None and n_files >= args.limit:
            break
        n_files += 1
        run_dir = os.path.dirname(path)
        method = os.path.basename(os.path.dirname(run_dir))
        if not method.startswith("jitrl"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                eps = json.load(fh)
        except Exception:
            continue
        if not isinstance(eps, list):
            continue
        task = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))

        for ep in eps:
            if not isinstance(ep, dict):
                continue
            y = bool(ep.get("success"))
            (steps_ok if y else steps_fail).append(int(ep.get("steps", 0)))
            term[str(ep.get("termination_reason"))] += 1
            for c in ep.get("chunks") or []:
                if not isinstance(c, dict):
                    continue
                n_chunks += 1
                s = c.get("state_summary")
                if s:
                    summaries[task].append(s)
                (chunk_ret_ok if y else chunk_ret_fail).append(float(c.get("return", 0.0)))
                nbrs = c.get("retrieved_neighbors") or []
                if nbrs:
                    ss = sorted((float(n.get("similarity", 0.0)) for n in nbrs), reverse=True)
                    per_chunk_sim.append((ss[0], ss[-1]))
                    for n in nbrs:
                        sims.append(float(n.get("similarity", 0.0)))
                        sim_ret.append((float(n.get("similarity", 0.0)), float(n.get("return", 0.0))))

    print("=" * 74)
    print("H1  键检索有没有区分度")
    print("=" * 74)

    # 1a 模板 token 占比
    tmpl_share, tok_sizes = [], []
    for task, ss in summaries.items():
        if len(ss) < 5:
            continue
        sets = [toks(x) for x in ss]
        freq: Counter = Counter()
        for st in sets:
            freq.update(st)
        common = {t for t, c in freq.items() if c >= 0.8 * len(sets)}
        avg_size = statistics.fmean(len(st) for st in sets)
        tok_sizes.append(avg_size)
        if avg_size:
            tmpl_share.append(len(common) / avg_size)
    if tmpl_share:
        print(f"  状态摘要 token 数（均值）        {statistics.fmean(tok_sizes):.1f}")
        print(f"  模板 token 占比（在 80% 状态里都出现的）"
              f" 均值 {statistics.fmean(tmpl_share):.1%}"
              f"  （中位 {statistics.median(tmpl_share):.1%}）")

    # 1b 任意两条状态的 Jaccard 分布
    jac: list[float] = []
    for task, ss in list(summaries.items()):
        if len(ss) < 8:
            continue
        sets = [toks(x) for x in ss[:60]]
        for i in range(len(sets)):
            for j in range(i + 1, min(len(sets), i + 8)):
                u = sets[i] | sets[j]
                if u:
                    jac.append(len(sets[i] & sets[j]) / len(u))
    if jac:
        print(f"  任意两条状态的 Jaccard         "
              f"均值 {statistics.fmean(jac):.3f}  "
              f"p10 {pct(jac,0.1):.3f}  p90 {pct(jac,0.9):.3f}  "
              f"极差 {max(jac)-min(jac):.3f}")

    # 1c 检索到的邻居相似度分布
    if sims:
        print(f"  检索邻居的 similarity          "
              f"均值 {statistics.fmean(sims):.3f}  "
              f"p10 {pct(sims,0.1):.3f}  p90 {pct(sims,0.9):.3f}")
    if per_chunk_sim:
        spread = [a - b for a, b in per_chunk_sim]
        print(f"  同一批 top-10 内部相似度极差     "
              f"均值 {statistics.fmean(spread):.3f}  "
              f"中位 {statistics.median(spread):.3f}")
        print(f"  ⇒ 极差越小 = 前 10 条分不出高下 = 排序近随机")

    print("\n" + "=" * 74)
    print("H3  存进去的分数有没有信息")
    print("=" * 74)
    if sim_ret:
        rho = spearman([a for a, _ in sim_ret], [b for _, b in sim_ret])
        print(f"  Spearman(相似度, 邻居 return)   "
              f"{rho:.3f}" if rho is not None else "  无法计算")
        # 高低相似度各取一端的 return 均值
        srt = sorted(sim_ret, key=lambda x: x[0])
        k = max(1, len(srt) // 4)
        lo = statistics.fmean(r for _, r in srt[:k])
        hi = statistics.fmean(r for _, r in srt[-k:])
        print(f"  低相似度 1/4 的 return 均值     {lo:.4f}")
        print(f"  高相似度 1/4 的 return 均值     {hi:.4f}"
              f"   （差 {hi-lo:+.4f}）")
    if chunk_ret_ok or chunk_ret_fail:
        mo = statistics.fmean(chunk_ret_ok) if chunk_ret_ok else 0.0
        mf = statistics.fmean(chunk_ret_fail) if chunk_ret_fail else 0.0
        print(f"\n  chunk return 均值  成功回合 {mo:.4f}"
              f" / 失败回合 {mf:.4f}   （差 {mo-mf:+.4f}）")
        print("  ⇒ 若两者几乎相同，说明写进记忆的这个分数分不出成败")

    print("\n" + "=" * 74)
    print("H2  VLA 底层执行能力")
    print("=" * 74)
    print(f"  termination_reason 分布：")
    for k, v in term.most_common(8):
        print(f"    {k:<34}{v:>7}")
    if steps_ok and steps_fail:
        print(f"  成功回合步数  中位 {statistics.median(steps_ok):.0f}"
              f"  p90 {pct(steps_ok,0.9):.0f}")
        print(f"  失败回合步数  中位 {statistics.median(steps_fail):.0f}"
              f"  p90 {pct(steps_fail,0.9):.0f}")
        print("  ⇒ 失败回合若明显更短，说明是在早期执行阶段就崩了（支持 H2）")

    print(f"\n  扫描 {n_files} 个 run / {n_chunks} 个决策步")
    return 0


if __name__ == "__main__":
    sys.exit(main())
