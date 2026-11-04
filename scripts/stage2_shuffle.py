#!/usr/bin/env python3
"""stage2_shuffle.py —— 打乱记忆池的判据（修正版）

修正两处：
  1) 自由候选模式下，候选与邻居的匹配是 action_similarity(normalized-token Jaccard)
     阈值匹配，不是字符串相等；固定九类模式才是按 semantic_key 精确分组。
  2) 未匹配到的候选 Q = 0（不是乐观 UCB 分支）。

流程：先扫阈值把 Q 的复现率打到高，再验证整条流水线，再做三种打乱。
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

TEMP = 0.8
BETA = 0.40
TOK = re.compile(r"[a-z0-9]+")


def norm_action(s):
    return " ".join(TOK.findall((s or "").lower()))


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


def is_free_mode(keys):
    return not any(k and "|" in k for k in keys)


def est(nbrs, keys, cand_texts, th):
    """返回 (V, Q, A, normA)。匹配不上 → Q = 0。"""
    if not nbrs:
        return 0.0, [0.0] * len(keys), [0.0] * len(keys), [0.0] * len(keys)
    rets = [n["return"] for n in nbrs]
    V = sum(rets) / len(rets)
    free = is_free_mode(keys)
    Q = []
    for k, txt in zip(keys, cand_texts):
        if free:
            m = [n["return"] for n in nbrs
                 if jacc(txt or k, n.get("action_text") or "") >= th]
        else:
            m = [n["return"] for n in nbrs if n.get("action_key") == k]
        Q.append(sum(m) / len(m) if m else 0.0)
    A = [q - V for q in Q]
    mx = max((abs(a) for a in A), default=0.0)
    normA = [(a / mx) if mx > 0 else 0.0 for a in A]
    return V, Q, A, normA


def pipeline(chunk, th):
    nbrs = chunk.get("retrieved_neighbors") or []
    cands = chunk.get("candidates") or []
    keys = [c.get("semantic_key") for c in cands]
    texts = [c.get("text") for c in cands]
    base, u = chunk.get("base_logits"), chunk.get("uniform")
    if not nbrs or any(k is None for k in keys) or not base or u is None:
        return None
    if len(base) != len(keys):
        return None
    _, _, _, normA = est(nbrs, keys, texts, th)
    bidx = choose(softmax([b / TEMP for b in base]), u)
    uidx = choose(softmax([(b + BETA * na) / TEMP for b, na in zip(base, normA)]), u)
    return bidx, uidx, normA


def iter_chunks(root):
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
        rch = []
        for ep in eps:
            for c in ep.get("chunks") or []:
                if isinstance(c, dict):
                    rch.append(c)
        yield rch


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    allruns = list(iter_chunks(root))
    chunks = [c for r in allruns for c in r if c.get("retrieved_neighbors")]

    # ---------- 阈值扫描 ----------
    print("=" * 74)
    print("第一步：从数据里反推匹配阈值（Q 的逐位复现率）")
    print("=" * 74)
    print(f"  {'阈值':>6}{'Q 逐位一致':>14}{'choice 一致':>14}{'样本':>9}")
    best, best_rate = None, -1
    for th in (0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.85):
        nq = ok_q = nch = ok_ch = 0
        for c in chunks[:6000]:
            cands = c.get("candidates") or []
            keys = [x.get("semantic_key") for x in cands]
            texts = [x.get("text") for x in cands]
            if any(k is None for k in keys):
                continue
            _, Q, _, _ = est(c["retrieved_neighbors"], keys, texts, th)
            recQ = (c.get("value_estimate") or {}).get("Q")
            if recQ and len(recQ) == len(Q):
                nq += 1
                ok_q += all(abs(a - b) < 1e-3 for a, b in zip(Q, recQ))
            g = pipeline(c, th)
            if g is not None:
                nch += 1
                ok_ch += (g[0] != g[1]) == bool(c.get("choice_changed"))
        rq = ok_q / nq if nq else 0
        print(f"  {th:>6.2f}{rq:>13.1%}{ok_ch/max(nch,1):>14.1%}{nq:>9}")
        if rq > best_rate:
            best, best_rate = th, rq
    TH = best
    print(f"\n  采用阈值 TH = {TH}")

    # ---------- 全量验证 ----------
    print("\n" + "=" * 74)
    print("第二步：全量流水线验证")
    print("=" * 74)
    n = okc = oki = okp = 0
    for c in chunks:
        g = pipeline(c, TH)
        if g is None:
            continue
        n += 1
        bidx, uidx, normA = g
        okc += (bidx != uidx) == bool(c.get("choice_changed"))
        oki += uidx == c.get("updated_choice_index")
        base = c.get("base_logits")
        upd = [b + BETA * na for b, na in zip(base, normA)]
        mine = softmax([x / TEMP for x in upd])
        rp = c.get("updated_probs")
        if rp and len(rp) == len(mine):
            okp += all(abs(a - b) < 5e-3 for a, b in zip(mine, rp))
    print(f"  chunk {n}")
    print(f"  changed 一致 {okc}（{okc/max(n,1):.1%}）")
    print(f"  choice_index 一致 {oki}（{oki/max(n,1):.1%}）")
    print(f"  概率分布一致 {okp}（{okp/max(n,1):.1%}）")

    # ---------- 打乱 ----------
    print("\n" + "=" * 74)
    print("第三步：打乱后选择还会不会被改动")
    print("=" * 74)
    rng = random.Random(20260920)
    cnt = {k: [0, 0] for k in ("V2", "V3", "V4", "V5")}
    # 选择的一致性：真实检索选的，与「随机检索」「基准」选的，有多大概率一样
    agree = defaultdict(int)
    agree_n = defaultdict(int)
    for rch in allruns:
        cand_chunks = [c for c in rch if c.get("retrieved_neighbors")]
        if not cand_chunks:
            continue
        pool = []
        seen = set()
        for c in rch:
            for nb in c.get("retrieved_neighbors") or []:
                if nb.get("id") not in seen:
                    seen.add(nb.get("id"))
                    pool.append(nb)
        for c in rch:
            nbrs = c.get("retrieved_neighbors") or []
            if not nbrs:
                continue
            g = pipeline(c, TH)
            if g is None:
                continue
            cnt["V2"][0] += 1
            cnt["V2"][1] += 1 if g[0] != g[1] else 0
            agree_n["real_vs_base"] += 1
            agree["real_vs_base"] += 1 if g[1] == g[0] else 0

            # V3 打乱 return 与邻居的绑定
            c3 = dict(c)
            nn = [dict(x) for x in nbrs]
            rs = [x["return"] for x in nn]
            rng.shuffle(rs)
            for x, r in zip(nn, rs):
                x["return"] = r
            c3["retrieved_neighbors"] = nn
            c3["base_logits"], c3["uniform"] = c["base_logits"], c["uniform"]
            g3 = pipeline(c3, TH)
            if g3 is not None:
                cnt["V3"][0] += 1
                cnt["V3"][1] += 1 if g3[0] != g3[1] else 0
                agree_n["real_vs_V3"] += 1
                agree["real_vs_V3"] += 1 if g3[1] == g[1] else 0

            # V4 换成同 run 另一 chunk 的检索邻居
            o = cand_chunks[rng.randrange(len(cand_chunks))]
            if len(o.get("retrieved_neighbors") or []) == len(nbrs):
                c4 = dict(c)
                c4["retrieved_neighbors"] = o["retrieved_neighbors"]
                g4 = pipeline(c4, TH)
                if g4 is not None:
                    cnt["V4"][0] += 1
                    cnt["V4"][1] += 1 if g4[0] != g4[1] else 0
                    agree_n["real_vs_V4"] += 1
                    agree["real_vs_V4"] += 1 if g4[1] == g[1] else 0

            # V5 从该 run 迄今出现过的条目里随机抽同样多条
            if len(pool) >= len(nbrs):
                c5 = dict(c)
                c5["retrieved_neighbors"] = rng.sample(pool, len(nbrs))
                g5 = pipeline(c5, TH)
                if g5 is not None:
                    cnt["V5"][0] += 1
                    cnt["V5"][1] += 1 if g5[0] != g5[1] else 0
                    agree_n["real_vs_V5"] += 1
                    agree["real_vs_V5"] += 1 if g5[1] == g[1] else 0

    labels = {
        "V2": "V2 真实（相似度检索）",
        "V3": "V3 打乱 return 绑定",
        "V4": "V4 换一套邻居",
        "V5": "V5 随机池抽取",
    }
    base = None
    print(f"  {'条件':<26}{'chunk':>8}{'改选率':>10}{'相对 V2':>10}")
    print("  " + "-" * 56)
    for k in ("V2", "V3", "V4", "V5"):
        tn, tc = cnt[k]
        if not tn:
            continue
        r = tc / tn
        if k == "V2":
            base = r
        rel = f"{r/base:.2f}×" if base and k != "V2" else "—"
        print(f"  {labels[k]:<26}{tn:>8}{r:>10.2%}{rel:>10}")

    print("\n" + "=" * 74)
    print("第四步：真实检索选出来的动作，与随机检索选出来的，有多大概率一样")
    print("=" * 74)
    print(f"  {'对比':<30}{'一致率':>10}{'样本':>9}")
    print("  " + "-" * 50)
    names = {
        "real_vs_base": "真实检索 vs 基准（记忆没改）",
        "real_vs_V3": "真实检索 vs 打乱 return 绑定",
        "real_vs_V4": "真实检索 vs 换成别的 chunk 的邻居",
        "real_vs_V5": "真实检索 vs 从池里随机抽",
    }
    for k in ("real_vs_base", "real_vs_V3", "real_vs_V4", "real_vs_V5"):
        m = agree_n.get(k, 0)
        if m:
            print(f"  {names[k]:<30}{agree[k]/m:>10.2%}{m:>9}")
    print("\n  判读：若「真实 vs 随机」的一致性 ≈ 「真实 vs 基准」的一致性，")
    print("        说明真实检索并不比随机更确定选择 ⇒ 检索没有携带信息。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
