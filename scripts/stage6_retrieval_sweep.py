#!/usr/bin/env python3
"""stage6_retrieval_sweep.py —— 检索侧的穷举扫描（全离线）

之前八种「条目值构造」全部没把状态条件化的占比推过 8.3%。
但那只测了「当前这 10 条的区分度」，没测「检索侧改动能不能提高区分度」。

这里扫三样真正动检索的：
  k      —— 只取相似度最高的前 k 条（trace 只存了 top-10，只能缩）
  w      —— Q 的聚合权重：不加权 / 按相似度 / 按相似度平方
  beta   —— 残差系数

对照条件（关键设计）：随机版不是「随便抽」，而是
  从同任务池里随机抽 k 条，**把真实 top-k 的相似度值按位置赋给它们**。
这样两边的相似度分布完全一致，只换「哪几条被选中」——
测的才是纯粹的「状态匹配有没有用」。
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import statistics as st
import sys

GAMMA = 0.95
TEMP = 0.8
TH = 0.6
EXPL = 0.05
UCB = 5.0
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


def normA_of(nbrs, cands, wmode):
    if not nbrs:
        return None
    if wmode == "u":
        w = [1.0] * len(nbrs)
    else:
        p = 1.0 if wmode == "s" else 2.0
        w = [max(float(n.get("similarity", 0.0)), 1e-9) ** p for n in nbrs]
    sw = sum(w)
    V = sum(wi * n["return"] for wi, n in zip(w, nbrs)) / sw
    free = not any("|" in (c.get("semantic_key") or "") for c in cands)
    Q = []
    for c in cands:
        key, text = c.get("semantic_key"), c.get("text")
        if free:
            idx = [i for i, n in enumerate(nbrs)
                   if jacc(text or key, n.get("action_text") or "") >= TH]
        else:
            idx = [i for i, n in enumerate(nbrs) if n.get("action_key") == key]
        if idx:
            s = sum(w[i] for i in idx)
            q = sum(w[i] * nbrs[i]["return"] for i in idx) / s
        elif c.get("exploration_uniform") is not None and c["exploration_uniform"] < EXPL:
            q = V + UCB / len(nbrs)
        else:
            q = 0.0
        Q.append(q)
    A = [q - V for q in Q]
    sc = max((abs(a) for a in A), default=0.0)
    return [(a / sc) if sc else 0.0 for a in A]


def pick(chunk, normA, beta):
    base, u = chunk.get("base_logits"), chunk.get("uniform")
    if normA is None or u is None or not base or len(base) != len(normA):
        return None
    b = choose(softmax([x / TEMP for x in base]), u)
    s = choose(softmax([(x + beta * a) / TEMP for x, a in zip(base, normA)]), u)
    return b, s


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    x2 = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(max(x2, 0.0) / 2))


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "**", "episodes.json"), recursive=True)):
        rd = os.path.dirname(path)
        if not os.path.basename(os.path.dirname(rd)).startswith("jitrl"):
            continue
        try:
            eps = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(eps, list):
            continue
        chs = [c for ep in eps for c in (ep.get("chunks") or []) if isinstance(c, dict)]
        if not chs:
            continue
        seen, pool = set(), []
        for c in chs:
            for n in c.get("retrieved_neighbors") or []:
                i = n.get("id")
                if i is None or i in seen:
                    continue
                seen.add(i)
                pool.append({"action_key": n.get("action_key"),
                             "action_text": n.get("action_text"),
                             "return": float(n.get("return", 0.0)),
                             "similarity": float(n.get("similarity", 0.0))})
        runs.append((chs, pool))

    KS = (1, 3, 5, 10)
    WS = ("u", "s", "s2")
    BS = (0.2, 0.4, 0.8, 1.6)
    conds = [(k, w, b) for k in KS for w in WS for b in BS]
    cnt = {c: {"r": [0, 0], "m": [0, 0], "p": []} for c in conds}

    rng = random.Random(20260920)
    for chs, pool in runs:
        if len(pool) < 10:
            continue
        for c in chs:
            nbrs = c.get("retrieved_neighbors")
            cands = c.get("candidates") or []
            if not nbrs or not cands:
                continue
            if any(x.get("semantic_key") is None for x in cands):
                continue
            srt = sorted(nbrs, key=lambda n: -float(n.get("similarity", 0.0)))
            for k in KS:
                if len(srt) < k:
                    continue
                top = srt[:k]
                sims = [float(n.get("similarity", 0.0)) for n in top]
                # 随机版：从池里抽 k 条，按位置赋上真实 top-k 的相似度
                samp = rng.sample(pool, k)
                fake = [{**s, "similarity": si} for s, si in zip(samp, sims)]
                for w in WS:
                    nr = normA_of(top, cands, w)
                    nm = normA_of(fake, cands, w)
                    if nr is None or nm is None:
                        continue
                    for b in BS:
                        key = (k, w, b)
                        g1 = pick(c, nr, b)
                        g2 = pick(c, nm, b)
                        if g1 is None or g2 is None:
                            continue
                        f1, f2 = g1[0] != g1[1], g2[0] != g2[1]
                        cnt[key]["r"][0] += 1
                        cnt[key]["r"][1] += f1
                        cnt[key]["m"][0] += 1
                        cnt[key]["m"][1] += f2
                        cnt[key]["p"].append((f1, f2))

    rows = []
    for key in conds:
        v = cnt[key]
        if not v["r"][0]:
            continue
        f1 = v["r"][1] / v["r"][0]
        f2 = v["m"][1] / v["m"][0]
        d = f1 - f2
        share = d / f1 if f1 else 0.0
        b_ = sum(1 for x, y in v["p"] if x and not y)
        c_ = sum(1 for x, y in v["p"] if y and not x)
        p = mcnemar(b_, c_)
        rows.append((key, f1, f2, d, share, p))

    rows.sort(key=lambda r: -r[4])
    print("=" * 96)
    print("检索侧扫描：k / 聚合权重 / beta（判据 = 真实检索 vs 同位置赋相似度的随机抽）")
    print("=" * 96)
    print(f"  {'k':>3}{'权重':>6}{'beta':>7}{'改选率(真实)':>14}{'改选率(随机)':>14}"
          f"{'状态条件化':>12}{'占比':>9}{'p':>11}")
    print("  " + "-" * 88)
    for (k, w, b), f1, f2, d, share, p in rows:
        mark = "  ←" if (p < 0.01 and share > 0.15) else ""
        print(f"  {k:>3}{w:>6}{b:>7}{f1:>14.2%}{f2:>14.2%}"
              f"{d:>+12.2%}{share:>9.1%}{p:>11.2e}{mark}")
    print("\n  对照：现状（k=10, 不加权, beta=0.4）的占比是 7.1%（p=1.1e-06）")
    print("  ← 标注的是：显著 且 占比 >15%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
