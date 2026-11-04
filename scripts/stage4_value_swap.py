#!/usr/bin/env python3
"""stage4_value_swap.py

两件事：

【A】量 memory 在 stop 候选上的占比
     stop 是九类动作之一，memory 对它也在算 advantage，但部署时被屏蔽
     （配置 environment_only_no_stop_v1）。量一下 13.18% 的改选里有多少落在 stop 上。

【B】换值探针（离线模拟，不是 rollout）
     把条目上挂的数从「Qwen 评估器打分的折扣回报」换成「环境终局成功的折扣值」，
     其余全冻，重算选择，看：
       - 换值后改选率变成多少
       - 换值后的选择与真实选择的一致率
       - 条目上两种值的秩相关与离散度
     ⚠ 这只能回答「换值会不会改变选择的结构」，不能回答「成功率会不会动」——
        后者必须真跑 rollout。
"""

from __future__ import annotations

import glob
import json
import math
import os
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
    return V, Q, A, [(a / sc) if sc else 0.0 for a in A]


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


def is_stop(c):
    k = (c.get("semantic_key") or "") + " " + (c.get("text") or "") + " " + (c.get("type") or "")
    return "stop" in k.lower()


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "artifacts"

    # ---------- A：stop 占比 ----------
    n_ch = 0
    n_has_stop_cand = 0
    n_flip = 0
    n_flip_to_stop = 0
    n_flip_from_stop = 0
    n_real_stop_chosen = 0

    # ---------- B：换值 ----------
    n_swap = 0
    n_swap_flip = 0
    agree_env_base = 0
    agree_env_real = 0
    pairs = []          # (Qwen return, env value) 逐条目
    seen_entry = set()

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

        succ, nch = {}, {}
        for ep in eps:
            i = ep.get("episode_index")
            succ[i] = 1.0 if ep.get("success") else 0.0
            nch[i] = len(ep.get("chunks") or [])

        for ep in eps:
            for c in ep.get("chunks") or []:
                if not isinstance(c, dict):
                    continue
                nbrs = c.get("retrieved_neighbors") or []
                cands = c.get("candidates") or []
                if not nbrs or not cands:
                    continue
                n_ch += 1
                if any(is_stop(x) for x in cands):
                    n_has_stop_cand += 1
                g = run(c)
                if g is None:
                    continue
                bidx, uidx = g
                flipped = bidx != uidx
                n_flip += flipped
                if any(is_stop(x) for x in cands):
                    if flipped and is_stop(cands[uidx]):
                        n_flip_to_stop += 1
                    if flipped and is_stop(cands[bidx]):
                        n_flip_from_stop += 1
                if is_stop(cands[uidx]):
                    n_real_stop_chosen += 1

                # --- 换值 ---
                swapped = []
                for n in nbrs:
                    e = n.get("episode")
                    t = n.get("chunk")
                    if e not in succ or e not in nch or t is None:
                        swapped = None
                        break
                    T = nch[e]
                    v = (GAMMA ** max(0, T - 1 - int(t))) * succ[e]
                    swapped.append({"action_key": n.get("action_key"),
                                    "action_text": n.get("action_text"),
                                    "return": v, "id": n.get("id")})
                    key = (os.path.relpath(rd, root), n.get("id"))
                    if key not in seen_entry:
                        seen_entry.add(key)
                        pairs.append((float(n.get("return", 0.0)), v))
                if not swapped:
                    continue
                g2 = run(c, nbrs=swapped)
                if g2 is None:
                    continue
                n_swap += 1
                n_swap_flip += g2[0] != g2[1]
                agree_env_base += g2[1] == g2[0]
                agree_env_real += g2[1] == uidx

    print("=" * 78)
    print("【A】memory 在 stop 候选上的占比")
    print("=" * 78)
    print(f"  有检索命中的决策步            {n_ch}")
    print(f"  候选集合里含 stop 的           {n_has_stop_cand}（{n_has_stop_cand/max(n_ch,1):.2%}）")
    print(f"  真实改选                      {n_flip}（{n_flip/max(n_ch,1):.2%}）")
    print(f"  真实最终选中 stop 的           {n_real_stop_chosen}（{n_real_stop_chosen/max(n_ch,1):.2%}）")
    print(f"  改选的方向落在 stop 上）        {n_flip_to_stop}"
          f"  占全部改选的 {n_flip_to_stop/max(n_flip,1):.2%}")
    print(f"  改选是「从 stop 改成别的」）     {n_flip_from_stop}"
          f"  占全部改选的 {n_flip_from_stop/max(n_flip,1):.2%}")

    print("\n" + "=" * 78)
    print("【B】换值探针（离线模拟，非 rollout）")
    print("=" * 78)
    print(f"  可换值的决策步                {n_swap}")
    if n_swap:
        print(f"  换值后的改选率                {n_swap_flip/n_swap:.2%}")
        print(f"  换值后的选择 == 基准的选择      {agree_env_base/n_swap:.2%}")
        print(f"  换值后的选择 == 真实检索的选择   {agree_env_real/n_swap:.2%}")
    print(f"\n  逐条目两种值的对照          {len(pairs)} 条")
    if pairs:
        q = [a for a, _ in pairs]
        e = [b for _, b in pairs]
        print(f"    Qwen 回报   均值 {st.fmean(q):.4f}  标准差 {st.pstdev(q):.4f}"
              f"  极差 {max(q)-min(q):.4f}")
        print(f"    环境值      均值 {st.fmean(e):.4f}  标准差 {st.pstdev(e):.4f}"
              f"  极差 {max(e)-min(e):.4f}")
        # Spearman
        n = len(pairs)
        def ranks(v):
            o = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
            while i < n:
                j = i
                while j + 1 < n and v[o[j+1]] == v[o[i]]:
                    j += 1
                avg = (i + j) / 2 + 1
                for k in range(i, j + 1):
                    r[o[k]] = avg
                i = j + 1
            return r
        rx, ry = ranks(q), ranks(e)
        mx, my = st.fmean(rx), st.fmean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
        dy = math.sqrt(sum((b - my) ** 2 for b in ry))
        rho = num / (dx * dy) if dx and dy else float("nan")
        print(f"    Spearman(Qwen 回报, 环境值) = {rho:.3f}")

    print("\n  ⚠ 这一步只回答「换值会不会改变选择的结构」。")
    print("     成功率会不会动，必须真跑 rollout 才能回答。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
