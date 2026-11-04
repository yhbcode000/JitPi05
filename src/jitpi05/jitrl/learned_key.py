"""阶段 2：把「检索键」本身变成可学的，梯度来自**环境回报**。

这是 resMem 那个问题的「照回报学」那一半。与 `JitRLMemory` 的区别**只有一处**：

    JitRLMemory          等权 Jaccard 给状态摘要算相似度，权重写死
    LearnedKeyMemory     给摘要的每个特征族一个可学权重 w_f，
                         相似度按 w_f 加权，并用**回合回报**对 w_f 做 REINFORCE

其余一律照搬：top-k 检索、按候选动作聚合回报、advantage 去均值、β 缩放、
均匀 base logits。所以两者的差别可以干净地归因到「检索键是不是学出来的」。

⚠ 两处必须记下来的实现选择：
  1. **选择用 softmax 采样而不是 argmax**。JitRL 的规则是在均匀 base logits 上
     加 β·advantage 后取 argmax，但 argmax 没有可用的对数概率，梯度写不出来。
     这里换成同温度的 softmax 采样，使 log π 良定义。这是**唯一**为可微性做的让步。
  2. **回合末更新一次**，不是每步更新——GRPO/REINFORCE 的常规做法，也贴合
     「一个回合只有一次反馈」的装置现实。
"""

from __future__ import annotations

from typing import Any

import torch

FEATURES = ("task", "x", "y", "z", "grip")


def _feat(summary: str) -> dict[str, str]:
    """把 `task=... x=0.10 y=-0.05 z=0.90 grip=0.04` 拆成特征字典。

    task 值里带空格，`split()` 会把它切碎——所以只取含 `=` 的那些 token，
    且 task 只保留第一个 token（任务在一次 run 内是常量，够用了）。
    """
    out: dict[str, str] = {}
    for tok in str(summary).split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k in FEATURES and k not in out:
            out[k] = v
    return out


class LearnedKeyMemory:
    """检索键可学、以环境回报为训练信号的记忆。"""

    def __init__(
        self,
        top_k: int = 10,
        lr: float = 5.0,
        beta: float = 0.40,
        temperature: float = 0.8,
        seed: int = 17,
    ) -> None:
        self.top_k = int(top_k)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self._gen = torch.Generator().manual_seed(int(seed))
        raw = torch.zeros(len(FEATURES), dtype=torch.float64)
        self.raw = torch.nn.Parameter(raw)
        self._opt = torch.optim.SGD([self.raw], lr=float(lr))
        self._entries: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []
        self.baseline = 0.0
        self.updates = 0
        self.last_grad_norm = 0.0

    # ---------------------------------------------------------------- 权重
    def __len__(self) -> int:
        return len(self._entries)

    def weights(self) -> torch.Tensor:
        """softplus 保证权重为正；初始 raw=0 ⇒ w=ln2，特征族等权。"""
        return torch.nn.functional.softplus(self.raw)

    # ---------------------------------------------------------------- 检索
    def _sim(self, a: dict[str, str], b: dict[str, str]) -> torch.Tensor:
        """加权 Jaccard：分子是「特征相同」的权重和，分母是「至少一边有」的权重和。

        ⚠ 必须用 stack().sum() 而不是往预分配张量里原地赋值——后者在
        requires_grad=False 的张量上会把计算图丢掉，REINFORCE 就退化成常数。
        """
        w = self.weights()
        den_terms, hit_terms = [], []
        for i, f in enumerate(FEATURES):
            av, bv = a.get(f), b.get(f)
            if av is None and bv is None:
                continue
            den_terms.append(w[i])
            if av is not None and av == bv:
                hit_terms.append(w[i])
        if not den_terms:
            return torch.zeros((), dtype=torch.float64)
        den = torch.stack(den_terms).sum()
        num = (torch.stack(hit_terms).sum() if hit_terms
               else torch.zeros((), dtype=torch.float64))
        return num / den.clamp_min(1e-9)

    def _topk(self, sf: dict[str, str]) -> list[tuple[dict[str, str], float, str]]:
        if not self._entries:
            return []
        scored = [(float(self._sim(sf, e["f"])), e) for e in self._entries]
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [(e["f"], float(e["return"]), e["action_key"]) for _, e in scored[: self.top_k]]

    def _logits(self, sf, ents, commands) -> torch.Tensor:
        """候选打分 = 按相似度把检索集的回报分给各候选，**在整个检索集上归一**。

        ⚠ 这里有一个必须记下来的设计选择。最自然写法是「每个候选内部按相似度加权平均」，
        但那样当检索到的条目回报相同时（例如全部成功），相似度在分子分母上互相抵消，
        权重 w 的梯度恒等于零——实测 |g| = 0.0000，什么都没学到。
        改成在**整个检索集**上归一之后，相似度决定「哪个动作分到多少权重」，
        w 才有梯度可走。这是把检索键变成可学参数的**必要条件**。
        """
        keys = list(commands)
        per_key: list[list[torch.Tensor]] = [[] for _ in keys]
        for ef, ret, ak in ents:
            if ak in keys:
                per_key[keys.index(ak)].append(self._sim(sf, ef) * float(ret))
        if not any(per_key):
            return torch.zeros(len(keys), dtype=torch.float64)
        zero = torch.zeros((), dtype=torch.float64)
        # ⚠ 不做归一化。归一化（无论按候选内还是按检索集）都会让 w 在
        # 「检索到的条目回报相同」或「相似度彼此相同」时整体约掉，梯度恒为零——
        # 实测 |g| = 0.0000。用相似度加权和，advantage 再去均值，w 才有梯度。
        score = torch.stack([torch.stack(v).sum() if v else zero for v in per_key])
        adv = score - score.mean()
        return (self.beta / max(self.temperature, 1e-6)) * adv

    # ---------------------------------------------------------------- 选择
    def select(self, summary: str, commands) -> tuple[str, int]:
        if not self._entries:
            return "", -1
        keys = list(commands)
        sf = _feat(summary)
        ents = self._topk(sf)
        with torch.no_grad():
            logits = self._logits(sf, ents, keys)
            probs = torch.softmax(logits, dim=-1)
        idx = int(torch.multinomial(probs.to(torch.float32), 1, generator=self._gen))
        self._pending.append({"s": summary, "ents": ents, "chosen": keys[idx]})
        return keys[idx], idx

    def note_uniform(self, summary: str, chosen: str) -> None:
        """记忆还空着时走均匀随机，也要记下来（否则第一个回合没有梯度）。"""
        self._pending.append({"s": summary, "ents": [], "chosen": chosen})

    # ---------------------------------------------------------------- 更新
    def append_episode(self, records, returns, episode_index: int) -> None:
        for i, (rec, ret) in enumerate(zip(records, returns)):
            self._entries.append(
                {"id": len(self._entries), "f": _feat(rec["state"]),
                 "return": float(ret), "action_key": rec["action_key"],
                 "episode_index": int(episode_index), "chunk_index": int(i)}
            )

    def finish_episode(self, episode_return: float, commands) -> dict[str, float]:
        """回合末：用 (回报 − 基线) 对检索键权重做一步 REINFORCE。"""
        out = {"steps": float(len(self._pending)), "advantage": 0.0, "grad_norm": 0.0}
        if not self._pending:
            return out
        advantage = float(episode_return) - self.baseline
        out["advantage"] = advantage
        keys = list(commands)
        loss = torch.zeros((), dtype=torch.float64)
        used = 0
        for rec in self._pending:
            # 跳过没有检索结果的步（第一个回合记忆是空的）——那里没有策略信息，
            # 强行纳入会让 loss 退化成常数，backward() 直接报「不 requires_grad」。
            if rec["chosen"] not in keys or not rec["ents"]:
                continue
            logits = self._logits(_feat(rec["s"]), rec["ents"], keys)
            logp = torch.log_softmax(logits, dim=-1)[keys.index(rec["chosen"])]
            loss = loss - logp
            used += 1
        if used == 0:
            self._pending.clear()
            return out
        loss = loss / used * advantage
        if not loss.requires_grad:          # 兜底：仍然没有图就不更新
            self._pending.clear()
            return out
        self._opt.zero_grad()
        loss.backward()
        g = float(self.raw.grad.norm()) if self.raw.grad is not None else 0.0
        self.last_grad_norm = g
        out["grad_norm"] = g
        self._opt.step()
        self.updates += 1
        self.baseline = 0.9 * self.baseline + 0.1 * float(episode_return)
        self._pending.clear()
        return out

    # ---------------------------------------------------------------- 存档
    def key_state(self) -> dict[str, Any]:
        """学到的检索键本身——单独一份，便于事后直接读。"""
        return {
            "weights": {f: float(v) for f, v in zip(FEATURES, self.weights())},
            "raw": [float(x) for x in self.raw.detach()],
            "baseline": float(self.baseline),
            "updates": int(self.updates),
            "entries": len(self._entries),
        }

    def snapshot(self) -> list[dict]:
        """⚠ 必须是 list——metrics 的 load_and_write_run_metrics 会校验。
        学到的键附在每个条目上，读 memory.json 就能看到。"""
        state = self.key_state()
        out = []
        for e in self._entries:
            out.append({
                "id": int(e["id"]),
                "state": "",                 # 特征已解析成字典，原文不再保留
                "action_key": str(e["action_key"]),
                "action_text": str(e["action_key"]),
                "return": float(e["return"]),
                "episode_index": int(e["episode_index"]),
                "chunk_index": int(e["chunk_index"]),
                "features": dict(e["f"]),
                "learned_key": state,
            })
        return out
