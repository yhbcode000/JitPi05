#!/usr/bin/env python3
"""stage1c_action_sequence.py —— 失败回合的动作序列长什么样

区分「VLA 执行不了选中的动作」与「高层选错了动作」：
  若失败回合里高层反复选同一个动作、而且它没用  → 选得没错，是执行不了
  若失败回合里高层不停换动作、原地打转          → 高层决策本身在乱走
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "artifacts"

    ok_div, fail_div = [], []
    ok_rep, fail_rep = [], []
    ok_attempts, fail_attempts = [], []
    fail_seq: Counter = Counter()      # 失败回合里出现过的动作序列（压缩后）
    ok_seq: Counter = Counter()
    fail_top_act: Counter = Counter()
    ok_top_act: Counter = Counter()
    n_ok = n_fail = 0
    plateau_steps: list[int] = []      # 第一次连续 3 步同动作出现在第几步（失败回合）

    for path in sorted(glob.glob(os.path.join(root, "**", "episodes.json"), recursive=True)):
        run_dir = os.path.dirname(path)
        if not os.path.basename(os.path.dirname(run_dir)).startswith("jitrl"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                eps = json.load(fh)
        except Exception:
            continue
        if not isinstance(eps, list):
            continue
        for ep in eps:
            if not isinstance(ep, dict):
                continue
            chunks = ep.get("chunks") or []
            if len(chunks) < 3:
                continue
            acts, att = [], []
            for c in chunks:
                if not isinstance(c, dict):
                    continue
                sc = c.get("selected_candidate") or {}
                # 固定九类工作集用 semantic_key；自由候选模式没有它，退回 type/text
                a = sc.get("semantic_key") or sc.get("type") or sc.get("text") or "?"
                acts.append(str(a))
                att.append(int(c.get("planner_attempts", 1) or 1))
            if not acts:
                continue

            div = len(set(acts)) / len(acts)
            rep = sum(1 for i in range(1, len(acts)) if acts[i] == acts[i - 1]) / (len(acts) - 1)
            # 压缩连续重复，得到「换了几次动作」
            comp, prev = [], None
            for a in acts:
                if a != prev:
                    comp.append(a)
                    prev = a
            key = "→".join(comp[:8])

            if ep.get("success"):
                n_ok += 1
                ok_div.append(div); ok_rep.append(rep)
                ok_attempts.append(statistics.fmean(att))
                ok_seq[key] += 1
                ok_top_act[acts[0]] += 1
            else:
                n_fail += 1
                fail_div.append(div); fail_rep.append(rep)
                fail_attempts.append(statistics.fmean(att))
                fail_seq[key] += 1
                fail_top_act[acts[0]] += 1
                # 第一次出现「连续三步同动作」
                run = 1
                for i in range(1, len(acts)):
                    run = run + 1 if acts[i] == acts[i - 1] else 1
                    if run >= 3:
                        plateau_steps.append(i)
                        break

    def show(label, a, b, fmt="{:.3f}"):
        if a and b:
            print(f"  {label:<28} 成功 {fmt.format(statistics.fmean(a))}"
                  f"   失败 {fmt.format(statistics.fmean(b))}")

    print("=" * 74)
    print(f"  成功回合 {n_ok} / 失败回合 {n_fail}")
    print("=" * 74)
    show("动作多样性 distinct/total", ok_div, fail_div)
    show("相邻重复率（黏住同一动作）", ok_rep, fail_rep)
    show("planner_attempts 均值", ok_attempts, fail_attempts)

    if plateau_steps:
        print(f"\n  失败回合里「连续 3 步选同一个动作」首次出现在第 "
              f"{statistics.median(plateau_steps):.0f} 步（中位），"
              f"共 {len(plateau_steps)} 个回合出现")

    print("\n  失败回合最常见的动作序列（压缩连续重复后，前 8 步）：")
    for k, v in fail_seq.most_common(6):
        print(f"    {v:>5}×  {k}")
    print("\n  成功回合最常见的动作序列：")
    for k, v in ok_seq.most_common(6):
        print(f"    {v:>5}×  {k}")

    print("\n  回合第一个动作的分布：")
    print(f"    成功  {dict(ok_top_act.most_common(5))}")
    print(f"    失败  {dict(fail_top_act.most_common(5))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
