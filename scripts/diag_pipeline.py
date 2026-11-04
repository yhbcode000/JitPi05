#!/usr/bin/env python3
"""逐项对撞：把记录值与我复现的 V/Q/A/normA/logits 摆在一起看"""
import glob
import json
import math
import os
from collections import defaultdict

TEMP = 0.8


def softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


shown = 0
stat = defaultdict(int)
for path in sorted(glob.glob("artifacts/**/episodes.json", recursive=True)):
    rd = os.path.dirname(path)
    if not os.path.basename(os.path.dirname(rd)).startswith("jitrl"):
        continue
    try:
        eps = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(eps, list):
        continue
    for ep in eps:
        for c in ep.get("chunks") or []:
            if not isinstance(c, dict):
                continue
            nbrs = c.get("retrieved_neighbors") or []
            if not nbrs:
                continue
            ve = c.get("value_estimate") or {}
            cands = c.get("candidates") or []
            keys = [x.get("semantic_key") for x in cands]
            if any(k is None for k in keys):
                continue

            rets = [n["return"] for n in nbrs]
            V = sum(rets) / len(rets)
            per = defaultdict(list)
            for n in nbrs:
                per[n["action_key"]].append(n["return"])
            recQ = ve.get("Q") or []
            recA = ve.get("A") or []
            recN = ve.get("normalized_A") or []

            # 记录值 vs 我的纯 mean-by-key（不做任何未见分支）
            plain = [sum(per[k]) / len(per[k]) if k in per else None for k in keys]
            stat["V_match"] += 1 if recQ and abs(V - ve.get("V", -999)) < 1e-6 else 0
            stat["n"] += 1
            # 记录里 Q=0 的位置 → 是不是就是没见到的 key
            unseen_zero = sum(1 for k, q in zip(keys, recQ)
                              if k not in per and abs(q) < 1e-12)
            unseen_tot = sum(1 for k in keys if k not in per)
            stat["unseen_tot"] += unseen_tot
            stat["unseen_is_zero"] += unseen_zero

            if shown < 3:
                shown += 1
                print("=" * 74)
                print("neighbors:", [(n["action_key"], round(n["return"], 4)) for n in nbrs])
                print("cand keys:", keys)
                print("  V      rec", ve.get("V"), " mine", round(V, 6))
                print("  Q      rec", [round(x, 4) for x in recQ])
                print("         mine(plain)", [round(x, 4) if x is not None else None for x in plain])
                print("  A      rec", [round(x, 4) for x in recA])
                print("  normA  rec", [round(x, 4) for x in recN])
                bl, ul = c.get("base_logits"), c.get("updated_logits")
                print("  base_logits   ", [round(x, 4) for x in bl])
                print("  updated_logits", [round(x, 4) for x in ul])
                sh = [round(u - b, 4) for u, b in zip(ul, bl)]
                print("  shift(=u-b)   ", sh)
                if recN:
                    mx = max(abs(x) for x in recN) or 1
                    print("  shift / normA", [round(s / (n / mx), 4) if n else None
                                              for s, n in zip(sh, recN)])
                bp, up = c.get("base_probs"), c.get("updated_probs")
                print("  base_probs    ", [round(x, 5) for x in bp] if bp else None)
                print("  softmax(bl/0.8)", [round(x, 5) for x in softmax([b / TEMP for b in bl])])
                print("  updated_probs ", [round(x, 5) for x in up] if up else None)
                print("  softmax(ul/0.8)", [round(x, 5) for x in softmax([b / TEMP for b in ul])])
                print("  uniform", c.get("uniform"), "base_idx", c.get("base_choice_index"),
                      "upd_idx", c.get("updated_choice_index"))

print("=" * 74)
print(f"V 一致率 {stat['V_match']}/{stat['n']}")
print(f"未见的候选动作 {stat['unseen_tot']} 个，其中记录 Q 恰为 0 的 {stat['unseen_is_zero']}")
