#!/usr/bin/env python3
"""stage5_value_zoo.py —— 条目值构造的穷举（全离线）

背景：stage4 发现把条目值换成「环境终局成功」会让状态条件化的贡献归零。
但那个构造太稀（大部分条目挂 0）。这里把所有能从现有 trace 构造出来的
值表都试一遍，看有没有哪一个能同时满足：
  (a) 记忆整体还有可测的影响
  (b) 状态条件化的贡献显著且占比高于现状的 7.4%

⚠ 一个此前漏掉的事实：条目上挂的是「奖励-to-go」（未来分数的γ折扣和），
   它天然偏袒靠前的 chunk（越早累计的未来奖励越多）。单步分数可以从
   trace 反推：r_t = R_t − γ·R_{t+1}

判据只有一个：真实检索 vs 同任务随机抽 的改选率差（配对 McNemar）。
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
from collections import defaultdict

GAMMA = 0.95
TEMP = 0.8
BETA = 0.40
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


def normA_of(nbrs, cands):
    V = st.fmean([n["return"] for n in nbrs]) if nbrs else 0.0
    free = not any("|" in (c.get("semantic_key") or "") for c in cands)
    Q = []
    for c in cands:
        key, text = c.get("semantic_key"), c.get("text")
        if free:
            m = [n["return"] for n in nbrs
                 if jacc(text or key, n.get("action_text") or "") >= TH]
        else:
            m = [n["return"] for n in nbrs if n.get("action_key") == key]
        if m:
            q = st.fmean(m)
        elif not nbrs:
            q = 0.0
        elif c.get("exploration_uniform") is not None and c["exploration_uniform"] < EXPL:
            q = V + UCB / len(nbrs)
        else:
            q = 0.0
        Q.append(q)
    A = [q - V for q in Q]
    sc = max((abs(a) for a in A), default=0.0)
    return [(a / sc) if sc else 0.0 for a in A]


def choice_with(chunk, nbrs, values):
    cands = chunk.get("candidates") or []
    base, u = chunk.get("base_logits"), chunk.get("uniform")
    if not nbrs or not cands or u is None or not base or len(base) != len(cands):
        return None
    if any(c.get("semantic_key") is None for c in cands):
        return None
    nb = [{**n, "return": values[n["_k"]]} for n in nbrs]
    na = normA_of(nb, cands)
    bidx = choose(softmax([b / TEMP for b in base]), u)
    uidx = choose(softmax([(b + BETA * x) / TEMP for b, x in zip(base, na)]), u)
    return bidx, uidx


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 0.0, 1.0
    x2 = (abs(b - c) - 1) ** 2 / n
    return x2, math.erfc(math.sqrt(max(x2, 0.0) / 2))


def spear(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")

    def rk(v):
        o = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[o[k]] = a
            i = j + 1
        return r
    rx, ry = rk(xs), rk(ys)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


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
        succ, nch, stepr, attempts = {}, {}, {}, {}
        R = {}
        for ep in eps:
            i = ep.get("episode_index")
            succ[i] = 1.0 if ep.get("success") else 0.0
            ch = ep.get("chunks") or []
            nch[i] = len(ch)
            for c in ch:
                t = c.get("chunk_index")
                if t is None:
                    continue
                R[(i, int(t))] = float(c.get("return", 0.0))
                attempts[(i, int(t))] = int(c.get("planner_attempts", 1) or 1)
        for (i, t), v in R.items():
            nxt = R.get((i, t + 1), 0.0)
            stepr[(i, t)] = v - GAMMA * nxt      # 反推单步分数
        ep_succ_rate = st.fmean(succ.values()) if succ else 0.0
        chs = [c for ep in eps for c in (ep.get("chunks") or []) if isinstance(c, dict)]
        if not chs:
            continue
        # 池：每个唯一条目一条记录
        pool, seen = [], set()
        for c in chs:
            for n in c.get("retrieved_neighbors") or []:
                i = n.get("id")
                if i is None or i in seen:
                    continue
                seen.add(i)
                e, t = n.get("episode"), n.get("chunk")
                e, t = (int(e) if e is not None else None), (int(t) if t is not None else None)
                rec = {"id": int(i), "action_key": n.get("action_key"),
                       "action_text": n.get("action_text"),
                       "rtg": float(n.get("return", 0.0))}
                rec["step"] = stepr.get((e, t), rec["rtg"])
                d = (GAMMA ** max(0, nch.get(e, 1) - 1 - (t or 0))) if e in nch else 0.0
                y = succ.get(e, 0.0)
                rec["env"] = d * y
                rec["env_c"] = d * (y - ep_succ_rate)
                rec["step_gate"] = rec["step"] * y
                rec["step_c"] = rec["step"] * (y - ep_succ_rate)
                rec["att"] = -math.log(1.0 + attempts.get((e, t), 1))
                pool.append(rec)
        sr = st.fmean([p["step"] for p in pool]) if pool else 0.0
        for p in pool:
            p["step_center"] = p["step"] - sr
        runs.append((chs, pool))

    CONSTRUCTIONS = ["rtg", "step", "env", "env_c", "step_gate",
                     "step_c", "step_center", "att"]

    # 全局唯一的键：run 索引 + entry id
    for ri, (chs, pool) in enumerate(runs):
        for p in pool:
            p["_k"] = (ri, p["id"])

    rng = random.Random(20260920)
    res = {c: {"real": [0, 0], "same": [0, 0], "pair": []} for c in CONSTRUCTIONS}
    vals_dump = {c: [] for c in CONSTRUCTIONS}
    credit_of = {}

    for ri, (chs, pool) in enumerate(runs):
        by_id = {p["id"]: p for p in pool}
        V = {cname: {p["_k"]: p[cname] for p in pool} for cname in CONSTRUCTIONS}
        for c in chs:
            nb = c.get("retrieved_neighbors")
            if not nb:
                continue
            seeded = []
            for n in nb:
                p = by_id.get(n.get("id"))
                if p is None:
                    seeded = None
                    break
                seeded.append({**n, "_k": p["_k"]})
            if not seeded:
                continue
            if len(pool) < len(seeded):
                continue
            rand = rng.sample(pool, len(seeded))
            for cname in CONSTRUCTIONS:
                vmap = V[cname]
                g1 = choice_with(c, seeded, vmap)
                g2 = choice_with(c, rand, vmap)
                if g1 is None or g2 is None:
                    continue
                f1, f2 = g1[0] != g1[1], g2[0] != g2[1]
                res[cname]["real"][0] += 1
                res[cname]["real"][1] += f1
                res[cname]["same"][0] += 1
                res[cname]["same"][1] += f2
                res[cname]["pair"].append((f1, f2))

    print("=" * 92)
    print("条目值构造的穷举（全部离线；判据 = 真实检索 vs 同任务随机抽）")
    print("=" * 92)
    print(f"  {'构造':<14}{'改选率(真实)':>13}{'改选率(随机)':>13}"
          f"{'状态条件化':>12}{'占比':>9}{'McNemar p':>12}")
    print("  " + "-" * 84)
    for cname in CONSTRUCTIONS:
        rr = res[cname]["real"]
        ss = res[cname]["same"]
        if not rr[0]:
            continue
        f1 = rr[1] / rr[0]
        f2 = ss[1] / ss[0]
        d = f1 - f2
        share = d / f1 if f1 else 0.0
        b = sum(1 for x, y in res[cname]["pair"] if x and not y)
        cc = sum(1 for x, y in res[cname]["pair"] if y and not x)
        _, p = mcnemar(b, cc)
        star = "  ←" if (p < 0.01 and share > 0.15) else ""
        print(f"  {cname:<14}{f1:>13.2%}{f2:>13.2%}{d:>+12.2%}{share:>9.1%}{p:>12.2e}{star}")

    print("\n  对照：现状（rtg）的状态条件化贡献 = 0.98pp，占 7.4%，p=1.9e-7")
    print("  标注 ← 的是：显著（p<0.01）且占比 >15% 的构造")
    return 0


if __name__ == "__main__":
    sys.exit(main())
