#!/usr/bin/env python3
"""stage2b_crosstask.py —— 跨任务随机池的收紧版

stage2 已证明：换成【同 run（同任务）内】随机抽的邻居，改选率 0.93×、
与真实检索的一致率 86.4%，而"记忆完全没改"的一致率是 86.8%。

但同任务的池仍带着该任务记忆的动作边缘分布，所以那一版只能支持
「状态条件化检索不比无条件抽取更确定选择」。这一版把池换成
【另一个任务】的记忆，做收紧：

  V2  真实检索（同任务、按相似度）
  V6  跨任务随机池（同实验同方法、不同任务的全部记忆）
  V7  本 run 最早出现的 10 条（确定性、与状态完全无关）

阈值 TH = 0.6（stage2 扫出来的），其余流水线与 stage2 完全一致。
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict

TEMP = 0.8
BETA = 0.40
TH = 0.6
TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(TOK.findall((s or "").lower()))


def jacc(a, b):
    ta, tb = toks(a), toks(b)
    u = ta | tb
    return len(ta & tb) / len(u) if u else 0.0


def softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def choose(probs, u):
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if c >= u:
            return i
    return len(probs) - 1


def est(nbrs, keys, texts):
    if not nbrs:
        return 0.0, [0.0] * len(keys), [0.0] * len(keys)
    V = statistics.fmean([n["return"] for n in nbrs])
    free = not any(k and "|" in k for k in keys)
    Q = []
    for k, txt in zip(keys, texts):
        if free:
            m = [n["return"] for n in nbrs
                 if jacc(txt or k, n.get("action_text") or "") >= TH]
        else:
            m = [n["return"] for n in nbrs if n.get("action_key") == k]
        Q.append(statistics.fmean(m) if m else 0.0)
    A = [q - V for q in Q]
    mx = max((abs(a) for a in A), default=0.0)
    return V, Q, [(a / mx) if mx > 0 else 0.0 for a in A]


def pipeline(chunk, nbrs=None):
    nb = chunk.get("retrieved_neighbors") or [] if nbrs is None else nbrs
    cands = chunk.get("candidates") or []
    keys = [c.get("semantic_key") for c in cands]
    texts = [c.get("text") for c in cands]
    base, u = chunk.get("base_logits"), chunk.get("uniform")
    if (not nb or any(k is None for k in keys) or not base or u is None
            or len(base) != len(keys)):
        return None
    _, _, normA = est(nb, keys, texts)
    bidx = choose(softmax([b / TEMP for b in base]), u)
    uidx = choose(softmax([(b + BETA * na) / TEMP for b, na in zip(base, normA)]), u)
    return bidx, uidx


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "artifacts"

    # ---- 收集：run 索引 + 每 (exp, method) 下按 task 分组的池 ----
    runs = []                                  # (run_key, panel, chunks, own_pool)
    groups: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(root, "**", "episodes.json"), recursive=True)):
        rd = os.path.dirname(path)
        method = os.path.basename(os.path.dirname(rd))
        if not method.startswith("jitrl"):
            continue
        task = os.path.basename(os.path.dirname(os.path.dirname(rd)))
        exp = os.path.relpath(rd, root).replace("\\", "/").split("/")[0]
        try:
            eps = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(eps, list):
            continue
        chs = [c for ep in eps for c in (ep.get("chunks") or []) if isinstance(c, dict)]
        if not chs:
            continue
        panel = ("libero_pro" if "libero_pro" in exp.lower()
                 else "libero_90" if "libero90" in exp.lower()
                 else "libero_10/标准四套" if ("libero10" in exp.lower() or "libero_10" in exp.lower())
                 else "其他")
        runs.append((os.path.relpath(rd, root), panel, chs))
        # 池：该 run 出现过的条目（去重、封顶 300 条省内存）
        seen, pool = set(), []
        for c in chs:
            for nb in c.get("retrieved_neighbors") or []:
                i = nb.get("id")
                if i not in seen:
                    seen.add(i)
                    pool.append({"action_key": nb.get("action_key"),
                                 "action_text": nb.get("action_text"),
                                 "return": nb.get("return"), "id": i})
                    if len(pool) >= 300:
                        break
            if len(pool) >= 300:
                break
        groups[(exp, method)][task].extend(pool)

    print("=" * 74)
    print("跨任务随机池的收紧版")
    print("=" * 74)
    rng = random.Random(20260920)
    cnt = {k: [0, 0] for k in ("V2", "V6", "V7")}
    agree = defaultdict(int)
    agree_n = defaultdict(int)
    per_panel = defaultdict(lambda: {k: [0, 0] for k in ("V2", "V6")})

    for run_key, panel, chs in runs:
        parts = run_key.replace("\\", "/").split("/")
        exp, task, method = parts[0], parts[-3], parts[-2]
        other = [p for (e, m), d in groups.items() if e == exp and m == method
                 for t, p in d.items() if t != task]
        other = [x for p in other for x in p]          # 跨任务合并池
        own = groups[(exp, method)].get(task, [])
        cand = [c for c in chs if c.get("retrieved_neighbors")]

        for c in cand:
            nbrs = c["retrieved_neighbors"]
            g = pipeline(c)
            if g is None:
                continue
            cnt["V2"][0] += 1
            flipped = g[0] != g[1]
            cnt["V2"][1] += flipped
            per_panel[panel]["V2"][0] += 1
            per_panel[panel]["V2"][1] += flipped
            agree_n["vs_base"] += 1
            agree["vs_base"] += 1 if g[1] == g[0] else 0

            # V6 跨任务随机池
            if len(other) >= len(nbrs):
                g6 = pipeline(c, nbrs=rng.sample(other, len(nbrs)))
                if g6 is not None:
                    cnt["V6"][0] += 1
                    f6 = g6[0] != g6[1]
                    cnt["V6"][1] += f6
                    per_panel[panel]["V6"][0] += 1
                    per_panel[panel]["V6"][1] += f6
                    agree_n["vs_V6"] += 1
                    agree["vs_V6"] += 1 if g6[1] == g[1] else 0

            # V7 本 run 最早的 10 条
            if len(own) >= len(nbrs):
                g7 = pipeline(c, nbrs=own[:len(nbrs)])
                if g7 is not None:
                    cnt["V7"][0] += 1
                    cnt["V7"][1] += 1 if g7[0] != g7[1] else 0
                    agree_n["vs_V7"] += 1
                    agree["vs_V7"] += 1 if g7[1] == g[1] else 0

    labels = {"V2": "V2 真实（同任务·相似度）",
              "V6": "V6 跨任务随机池",
              "V7": "V7 本 run 最早 10 条"}
    print(f"  {'条件':<28}{'chunk':>8}{'改选率':>10}{'相对 V2':>10}")
    print("  " + "-" * 58)
    base = None
    for k in ("V2", "V6", "V7"):
        n, c_ = cnt[k]
        if not n:
            continue
        r = c_ / n
        if k == "V2":
            base = r
        rel = f"{r/base:.2f}×" if base and k != "V2" else "—"
        print(f"  {labels[k]:<28}{n:>8}{r:>10.2%}{rel:>10}")

    print("\n  选择一致率（真实检索选出的动作，与下面这些有多大概率一样）：")
    for k, nm in (("vs_base", "vs 基准（记忆完全没改）"),
                  ("vs_V6", "vs 跨任务随机池"),
                  ("vs_V7", "vs 本 run 最早 10 条")):
        m = agree_n.get(k, 0)
        if m:
            print(f"    {nm:<26}{agree[k]/m:>10.2%}   (n={m})")

    print("\n  分面板（改选率）：")
    for p in ("libero_10/标准四套", "libero_90", "libero_pro"):
        d = per_panel.get(p)
        if not d or not d["V2"][0]:
            continue
        r2 = d["V2"][1] / d["V2"][0]
        s = f"    {p:<20} V2 {r2:>7.2%} (n={d['V2'][0]})"
        if d["V6"][0]:
            r6 = d["V6"][1] / d["V6"][0]
            s += f"   V6 {r6:>7.2%}   相对 {r6/r2 if r2 else 0:.2f}×"
        print(s)

    print("\n  判读：若 V6/V7 的改选率与 V2 相当、且「真实 vs 随机」的一致性 ≈ 「真实 vs 基准」，")
    print("        则连『状态条件化』这一层也没有贡献 ⇒ 检索携带的信息量 ≈ 0。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
