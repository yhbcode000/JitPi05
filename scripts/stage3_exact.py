#!/usr/bin/env python3
"""stage3_exact.py —— 按源码规则精确重建 + 配对显著性检验

与 stage2 / stage2b 的差别：
  1) 匹配阈值取自源码常量 JITRL_FREE_ACTION_SIM_THRESHOLD = 0.6（不是从数据反推）
  2) 未匹配候选的乐观分支按源码规则重建：exploration_uniform < 0.05 时
     Q = V + 5.0/|N|，否则 Q = 0。uniform 直接读 trace 里记录的值
  3) V7 按条目 id 排序取最小的 10 条（stage2b 按遍历顺序取，不一定是"最早"）
  4) 每个对照都做 McNemar 配对检验，判断差异是否显著

目标：把 Q 的逐位复现率从 93.5% 推到 ~100%。到了，后面所有数字才是源码级可信的。
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

TEMP = 0.8
BETA = 0.40
TH = 0.6          # JITRL_FREE_ACTION_SIM_THRESHOLD
EXPL = 0.05       # JITRL_EXPLORATION_RATE
UCB = 5.0         # JITRL_UCB_ALPHA
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
    """按源码规则重建 V / Q / A / normalized_A"""
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
    scale = max((abs(a) for a in A), default=0.0)
    return V, Q, A, [(a / scale) if scale else 0.0 for a in A]


def run(chunk, nbrs=None):
    nb = chunk.get("retrieved_neighbors") if nbrs is None else nbrs
    cands = chunk.get("candidates") or []
    base, u = chunk.get("base_logits"), chunk.get("uniform")
    if not nb or not cands or u is None or not base or len(base) != len(cands):
        return None
    if any(c.get("semantic_key") is None for c in cands):
        return None
    _, _, _, normA = est(nb, cands)
    bidx = choose(softmax([b / TEMP for b in base]), u)
    uidx = choose(softmax([(b + BETA * na) / TEMP for b, na in zip(base, normA)]), u)
    return bidx, uidx


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    return chi2, math.erfc(math.sqrt(max(chi2, 0.0) / 2))


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "artifacts"

    runs = []                                  # (exp, method, task, chunks, pool)
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
        seen, pool = set(), []
        for c in chs:
            for n in c.get("retrieved_neighbors") or []:
                i = n.get("id")
                if i not in seen and i is not None:
                    seen.add(i)
                    pool.append({"id": int(i), "action_key": n.get("action_key"),
                                 "action_text": n.get("action_text"),
                                 "return": n.get("return")})
        pool.sort(key=lambda x: x["id"])       # V7 要"最早"，按 id 排
        runs.append((exp, method, task, chs, pool))

    groups: dict = defaultdict(lambda: defaultdict(list))
    for exp, method, task, chs, pool in runs:
        groups[(exp, method)][task].extend(pool)

    chunks = [c for *_, chs, _p in [(r[0], r[1], r[2], r[3], r[4]) for r in runs]
              for c in chs if c.get("retrieved_neighbors")]

    # ---------- 一、精确复现 ----------
    n = nq = okq = okch = okidx = 0
    for c in chunks:
        cands = c.get("candidates") or []
        if not cands or any(x.get("semantic_key") is None for x in cands):
            continue
        _, Q, _, _ = est(c["retrieved_neighbors"], cands)
        rq = (c.get("value_estimate") or {}).get("Q")
        if rq and len(rq) == len(Q):
            nq += 1
            okq += all(abs(a - b) < 1e-3 for a, b in zip(Q, rq))
        g = run(c)
        if g:
            n += 1
            okch += (g[0] != g[1]) == bool(c.get("choice_changed"))
            okidx += g[1] == c.get("updated_choice_index")

    print("=" * 78)
    print("一、按源码规则精确重建（阈值 0.6 + 乐观分支用 trace 里的 uniform）")
    print("=" * 78)
    print(f"  Q 逐位一致        {okq}/{nq} = {okq/max(nq,1):.2%}")
    print(f"  changed 一致      {okch}/{n} = {okch/max(n,1):.2%}")
    print(f"  choice_index 一致 {okidx}/{n} = {okidx/max(n,1):.2%}")

    # ---------- 二、打乱 + McNemar ----------
    print("\n" + "=" * 78)
    print("二、三个对照（V7 已按条目 id 排序取最早）")
    print("=" * 78)
    rng = random.Random(20260920)
    cross = defaultdict(list)      # (V2f, V6f) 仅两者都算得出的 chunk
    earliest = defaultdict(list)   # (V2f, V7f)
    randpool = defaultdict(list)   # (V2f, V5f)
    cnt = {k: [0, 0] for k in ("V2", "V5", "V6", "V7")}

    for exp, method, task, chs, pool in runs:
        other = [x for t, p in groups[(exp, method)].items() if t != task for x in p]
        for c in chs:
            nb = c.get("retrieved_neighbors")
            if not nb:
                continue
            g = run(c)
            if not g:
                continue
            real = g[0] != g[1]
            cnt["V2"][0] += 1
            cnt["V2"][1] += real

            if len(other) >= len(nb):
                g6 = run(c, nbrs=rng.sample(other, len(nb)))
                if g6:
                    f6 = g6[0] != g6[1]
                    cnt["V6"][0] += 1
                    cnt["V6"][1] += f6
                    cross["pair"].append((real, f6))
            if len(pool) >= len(nb):
                g7 = run(c, nbrs=pool[:len(nb)])
                if g7:
                    f7 = g7[0] != g7[1]
                    cnt["V7"][0] += 1
                    cnt["V7"][1] += f7
                    earliest["pair"].append((real, f7))
                g5 = run(c, nbrs=rng.sample(pool, len(nb)))
                if g5:
                    f5 = g5[0] != g5[1]
                    cnt["V5"][0] += 1
                    cnt["V5"][1] += f5
                    randpool["pair"].append((real, f5))

    labels = {"V2": "V2 真实（同任务·相似度检索）",
              "V5": "V5 同任务池随机抽",
              "V6": "V6 跨任务随机池",
              "V7": "V7 本 run 最早 10 条（按 id）"}
    base = None
    print(f"  {'条件':<30}{'chunk':>8}{'改选率':>10}{'相对 V2':>10}")
    print("  " + "-" * 60)
    for k in ("V2", "V5", "V6", "V7"):
        nn, cc = cnt[k]
        if not nn:
            continue
        r = cc / nn
        if k == "V2":
            base = r
        print(f"  {labels[k]:<30}{nn:>8}{r:>10.2%}"
              f"{(f'{r/base:.2f}×' if base and k != 'V2' else '—'):>10}")

    print("\n  McNemar 配对检验（H0：真实检索与打乱对选择的改动没有系统差异）")
    print(f"  {'对比':<34}{'b':>7}{'c':>7}{'chi2':>9}{'p':>10}")
    print("  " + "-" * 68)
    for name, store in (("V2 vs V6 跨任务池", cross),
                        ("V2 vs V5 同任务随机抽", randpool),
                        ("V2 vs V7 最早10条", earliest)):
        pr = store.get("pair") or []
        if not pr:
            continue
        b = sum(1 for x, y in pr if x and not y)   # 真改、打乱不改
        cc2 = sum(1 for x, y in pr if y and not x)  # 打乱改、真不改
        chi2, p = mcnemar(b, cc2)
        print(f"  {name:<34}{b:>7}{cc2:>7}{chi2:>9.2f}{p:>10.2e}")
    print("\n  b > c 且显著 ⇒ 真实检索比打乱更常改动选择（检索有因果作用）")
    print("  b < c 且显著 ⇒ 真实检索反而更少改动选择")
    print("  b ≈ c       ⇒ 两者无差异")
    return 0


if __name__ == "__main__":
    sys.exit(main())
