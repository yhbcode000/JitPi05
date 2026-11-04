#!/usr/bin/env python3
"""stage4b_value_decomp.py —— 换值版的打乱分解

stage4 已量出：把条目上挂的数从「Qwen 折扣回报」换成「环境终局成功的折扣值」，
改选率从 13.18% 降到 5.92%。

现在问：这 5.92% 里，有多少是状态条件化、多少是记忆池结构？
若环境值下状态条件化的占比显著高于原来的 7.4%，说明「换 ④」能提高状态的权重。

五个条件，全在环境值下重算，再与「真实值」对照：
  真实值 + 真实检索        （= 现状）
  环境值 + 真实检索        （换值，不换检索）
  环境值 + 同任务随机抽
  环境值 + 本 run 最早十条（按 id）
  环境值 + 跨任务随机池
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


def est(nbrs, cands):
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


def run(chunk, nbrs, use_env: bool):
    cands = chunk.get("candidates") or []
    base, u = chunk.get("base_logits"), chunk.get("uniform")
    if not nbrs or not cands or u is None or not base or len(base) != len(cands):
        return None
    if any(c.get("semantic_key") is None for c in cands):
        return None
    nb = ([{**n, "return": n["env_value"]} for n in nbrs] if use_env else nbrs)
    normA = est(nb, cands)
    bidx = choose(softmax([b / TEMP for b in base]), u)
    uidx = choose(softmax([(b + BETA * na) / TEMP for b, na in zip(base, normA)]), u)
    return bidx, uidx


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 0.0, 1.0
    x2 = (abs(b - c) - 1) ** 2 / n
    return x2, math.erfc(math.sqrt(max(x2, 0.0) / 2))


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    runs = []
    groups: dict = defaultdict(lambda: defaultdict(list))

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
        succ, nch = {}, {}
        for ep in eps:
            i = ep.get("episode_index")
            succ[i] = 1.0 if ep.get("success") else 0.0
            nch[i] = len(ep.get("chunks") or [])
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
                e, t = n.get("episode"), n.get("chunk")
                if e in succ and e in nch and t is not None:
                    ev = (GAMMA ** max(0, nch[e] - 1 - int(t))) * succ[e]
                else:
                    ev = float(n.get("return", 0.0))
                pool.append({"id": int(i), "action_key": n.get("action_key"),
                             "action_text": n.get("action_text"),
                             "return": float(n.get("return", 0.0)),
                             "env_value": ev})
        pool.sort(key=lambda x: x["id"])
        runs.append((exp, method, task, chs, pool))
        groups[(exp, method)][task].extend(pool)

    rng = random.Random(20260920)
    cnt = {k: [0, 0] for k in ("real_real", "env_real", "env_same", "env_early", "env_cross")}
    pair_same, pair_cross, pair_early = [], [], []

    for exp, method, task, chs, pool in runs:
        other = [x for t, p in groups[(exp, method)].items() if t != task for x in p]
        env_by_id = {p["id"]: p["env_value"] for p in pool}
        for c in chs:
            nb = c.get("retrieved_neighbors")
            if not nb:
                continue
            nb_full = []
            for n in nb:
                ev = env_by_id.get(n.get("id"))
                if ev is None:
                    nb_full = None
                    break
                nb_full.append({**n, "env_value": ev})
            if not nb_full:
                continue

            g1 = run(c, nb_full, use_env=False)
            g2 = run(c, nb_full, use_env=True)
            if g1 is None or g2 is None:
                continue
            cnt["real_real"][0] += 1
            cnt["real_real"][1] += g1[0] != g1[1]
            r1 = g1[0] != g1[1]
            cnt["env_real"][0] += 1
            r2 = g2[0] != g2[1]
            cnt["env_real"][1] += r2

            if len(pool) >= len(nb_full):
                g3 = run(c, rng.sample(pool, len(nb_full)), use_env=True)
                if g3 is not None:
                    f3 = g3[0] != g3[1]
                    cnt["env_same"][0] += 1
                    cnt["env_same"][1] += f3
                    pair_same.append((r2, f3))
                g4 = run(c, pool[:len(nb_full)], use_env=True)
                if g4 is not None:
                    f4 = g4[0] != g4[1]
                    cnt["env_early"][0] += 1
                    cnt["env_early"][1] += f4
                    pair_early.append((r2, f4))
            if len(other) >= len(nb_full):
                g5 = run(c, rng.sample(other, len(nb_full)), use_env=True)
                if g5 is not None:
                    f5 = g5[0] != g5[1]
                    cnt["env_cross"][0] += 1
                    cnt["env_cross"][1] += f5
                    pair_cross.append((r2, f5))

    labels = {
        "real_real": "真实值 + 真实检索（现状）",
        "env_real": "环境值 + 真实检索（换值）",
        "env_same": "环境值 + 同任务随机抽",
        "env_early": "环境值 + 最早十条（按 id）",
        "env_cross": "环境值 + 跨任务随机池",
    }
    print("=" * 80)
    print("换值版的打乱分解")
    print("=" * 80)
    print(f"  {'条件':<34}{'chunk':>8}{'改选率':>10}")
    print("  " + "-" * 54)
    base = None
    for k in ("real_real", "env_real", "env_same", "env_early", "env_cross"):
        n, cc = cnt[k]
        if not n:
            continue
        r = cc / n
        if k == "env_real":
            base = r
        print(f"  {labels[k]:<34}{n:>8}{r:>10.2%}"
              f"{(f'（相对换值 {r/base:.2f}×）' if base and k.startswith('env_') and k != 'env_real' else '')}")

    print("\n  分解（全部在环境值下）：")
    if cnt["env_real"][0]:
        real = cnt["env_real"][1] / cnt["env_real"][0]
        same = cnt["env_same"][1] / max(cnt["env_same"][0], 1)
        cross = cnt["env_cross"][1] / max(cnt["env_cross"][0], 1)
        print(f"    环境值下总改选率              {real:.2%}")
        print(f"    同任务随机抽                  {same:.2%}"
              f"   （池结构贡献 {same:.2%}）")
        print(f"    状态条件化的边际贡献           {real - same:+.2%}"
              f"   占全部效应的 {(real-same)/real:.1%}" if real else "")
        print(f"    跨任务池                      {cross:.2%}")

    print("\n  McNemar（在环境值下，真实检索 vs 打乱）：")
    print(f"  {'对比':<30}{'b':>7}{'c':>7}{'chi2':>9}{'p':>11}")
    print("  " + "-" * 64)
    for name, pr in (("env 真实 vs 同任务随机抽", pair_same),
                     ("env 真实 vs 最早十条", pair_early),
                     ("env 真实 vs 跨任务池", pair_cross)):
        if not pr:
            continue
        b = sum(1 for x, y in pr if x and not y)
        c_ = sum(1 for x, y in pr if y and not x)
        x2, p = mcnemar(b, c_)
        print(f"  {name:<30}{b:>7}{c_:>7}{x2:>9.2f}{p:>11.2e}")

    print("\n  对照：真实值下的分解是 13.18% / 12.20% / 11.32% / 5.53%，")
    print("        状态条件化贡献 0.98pp（占 7.4%）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
