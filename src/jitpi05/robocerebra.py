"""RoboCerebra 任务套件动态注册。

RoboCerebra（arXiv:2506.06677，NeurIPS 2025 D&B）是一个长时程操作基准，
它的场景与物品**不属于 LIBERO-10**（COFFEE_TABLESCENE / KITCHEN_TABLESCENE
下的 cream cheese、popcorn、butter、red coffee mug、milk 等），但 bddl 与初始
状态都是 **LIBERO/robosuite 的原生格式**，因此可以照 ``libero_pro`` 的同一套
机制注册，低层策略不必更换。

与 LIBERO 原生 suite 的两处差别，注册时必须处理：

1. **初始状态格式**：RoboCerebra 的 ``.init`` 是形状 ``(D,)`` 的 numpy 向量，
   即**一个状态**；而 LIBERO 的 ``.pruned_init`` 是 ``(N, D)`` 的数组列表。
   LeRobot 侧按 ``init_states[init_state_id % len(init_states)]`` 取值，所以
   这里重排成 ``(1, D)`` —— ``% 1 = 0``，每个 case 的每个回合都从同一个状态
   出发。这**不是缺陷**：同起点意味着记忆开/关的比较是干净的配对干预，差异
   只能来自记忆本身。代价是一个 case 只有一个有效样本。

2. **任务名会撞**：60 个 case 只有 15 个唯一 bddl 名（同一任务在 Ideal /
   Observation_Mismatching / Random_Disturbance 下复用）。所以任务名一律带
   类型前缀。

覆盖情况：``Random_Disturbance`` 没有 init 文件，``Mix`` 只有 5 个，因此可用
case 数是 45 而不是 60。
"""

from __future__ import annotations

import os
import pickle
import re
import shutil
from pathlib import Path

import numpy as np

from libero.libero import benchmark as _benchmark
from libero.libero import get_libero_path
from libero.libero.benchmark import Benchmark, Task, register_benchmark
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

# 任务类型 → suite 名。顺序即评测顺序。
TASK_TYPES: dict[str, str] = {
    "Ideal": "robocerebra_ideal",
    "Memory_Execution": "robocerebra_memory_execution",
    "Memory_Exploration": "robocerebra_memory_exploration",
    "Observation_Mismatching": "robocerebra_observation_mismatching",
    "Mix": "robocerebra_mix",
}

# RoboCerebra 基准根目录。默认取环境变量，其次取 ~/datasets/RoboCerebra。
DEFAULT_ROOT = Path.home() / "datasets" / "RoboCerebra"


def bench_root() -> Path:
    """返回 RoboCerebra 基准根目录（内含 RoboCerebraBench/）。"""
    raw = os.environ.get("ROBOCEREBRA_BENCH_ROOT")
    return Path(raw) if raw else DEFAULT_ROOT


_registered = False
_registered_suites: tuple[str, ...] = ()


def _language_of(bddl_path: Path) -> str:
    """从 bddl 的 ``(:language ...)`` 取总任务描述，并把下划线还原成空格。"""
    text = bddl_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\(:language\s+([^)]*)\)", text)
    if not m:
        return bddl_path.stem.replace("_", " ")
    return " ".join(m.group(1).strip().split("_"))


def _case_ids(task_type: str) -> list[str]:
    """列出某类型下所有「有 init 文件」的 case，按数字序返回。"""
    root = bench_root() / "RoboCerebraBench"
    init_dir = root / "init_files" / task_type
    if not init_dir.is_dir():
        return []
    out = []
    for f in init_dir.glob("case*.init"):
        m = re.fullmatch(r"case(\d+)", f.stem)
        if m:
            out.append(f.stem)
    out.sort(key=lambda s: int(s[4:]))
    return out


def _register_suite_class(suite: str) -> None:
    """注册一个与 LIBERO 原生 suite 等价的 benchmark 类。"""

    def __init__(self, task_order_index: int = 0) -> None:
        Benchmark.__init__(self, task_order_index=task_order_index)
        self.name = suite
        self._make_benchmark()

    cls = type(suite.upper(), (Benchmark,), {"__init__": __init__})
    register_benchmark(cls)


def register_robocerebra_suites(
    types: tuple[str, ...] | None = None,
    max_cases: int | None = None,
) -> tuple[str, ...]:
    """把 RoboCerebra 的任务注册进 libero benchmark（幂等）。

    Args:
        types: 要注册的类型子集，默认全部。
        max_cases: 每个类型最多注册几个 case（用于先做小规模冒烟）。

    Returns:
        本次实际注册的 suite 名元组。
    """
    global _registered, _registered_suites

    wanted = tuple(types) if types else tuple(TASK_TYPES)
    for t in wanted:
        if t not in TASK_TYPES:
            raise ValueError(f"unknown RoboCerebra task type {t!r}; known: {tuple(TASK_TYPES)}")

    root = bench_root() / "RoboCerebraBench"
    if not root.is_dir():
        raise FileNotFoundError(
            f"RoboCerebraBench not found at {root}. "
            "Set ROBOCEREBRA_BENCH_ROOT or run the snapshot download."
        )

    bddl_root = Path(get_libero_path("bddl_files"))
    init_root = Path(get_libero_path("init_states"))

    for task_type in wanted:
        suite = TASK_TYPES[task_type]
        cases = _case_ids(task_type)
        if max_cases is not None:
            cases = cases[:max_cases]
        if not cases:
            continue

        (bddl_root / suite).mkdir(parents=True, exist_ok=True)
        (init_root / suite).mkdir(parents=True, exist_ok=True)

        task_names: list[str] = []
        _benchmark.task_maps.setdefault(suite, {})
        for case in cases:
            task = f"{task_type}_{case}"
            src_dir = root / task_type / case
            bddls = sorted(src_dir.glob("*.bddl"))
            if len(bddls) != 1:
                raise RuntimeError(f"{task}: expected exactly 1 bddl, found {len(bddls)}")
            src_bddl = bddls[0]

            dst_bddl = bddl_root / suite / f"{task}.bddl"
            if dst_bddl.is_symlink() or dst_bddl.exists():
                dst_bddl.unlink()
            dst_bddl.symlink_to(src_bddl)

            # init：(D,) -> (1, D)。LeRobot 侧按 len() 取模，1 会让每个回合
            # 都落在同一个初始状态上——这是有意的，见模块 docstring。
            src_init = root / "init_files" / task_type / f"{case}.init"
            with open(src_init, "rb") as fh:
                state = np.asarray(pickle.load(fh), dtype=np.float64).reshape(1, -1)
            dst_init = init_root / suite / f"{task}.pruned_init"
            # 必须用 torch.save：LIBERO/LeRobot 侧统一走 torch.load，而它先按
            # torch 自己的容器格式探测魔数，原始 pickle 会在那一步就失败。
            import torch

            torch.save(state, dst_init)

            _benchmark.task_maps[suite][task] = Task(
                name=task,
                language=_language_of(src_bddl),
                problem="Libero",
                problem_folder=suite,
                bddl_file=f"{task}.bddl",
                init_states_file=f"{task}.pruned_init",
            )
            task_names.append(task)

        libero_task_map[suite] = task_names
        if suite not in _benchmark.BENCHMARK_MAPPING:
            _register_suite_class(suite)

    _registered_suites = tuple(TASK_TYPES[t] for t in wanted if TASK_TYPES[t] in libero_task_map)
    _registered = True
    return _registered_suites


def _oracle_subtasks(task_type: str, case: str) -> list[str]:
    """oracle 条件用的子任务序列；``JITPI05_ORACLE_HOLD_FIRST=1`` 时只留第一条。"""
    import os as _os

    seq = ground_truth_subtasks(task_type, case)
    hold = _os.environ.get("JITPI05_ORACLE_HOLD_FIRST", "").strip().lower()
    if hold in ("1", "true", "yes", "on") and seq:
        return seq[:1]
    return seq


def robocerebra_task_specs(
    types: tuple[str, ...] | None = None,
    max_cases: int | None = None,
    max_steps: int = 2000,
) -> tuple[dict, ...]:
    """生成与 config 的 JITRL_TASKS 同构的任务描述。"""
    wanted = tuple(types) if types else tuple(TASK_TYPES)
    specs: list[dict] = []
    for task_type in wanted:
        suite = TASK_TYPES[task_type]
        cases = _case_ids(task_type)
        if max_cases is not None:
            cases = cases[:max_cases]
        for case in cases:
            specs.append(
                {
                    "name": f"{task_type}_{case}",
                    "suite": suite,
                    "task_id": cases.index(case),
                    "description": "",
                    "max_steps": max_steps,
                    "zero_shot": True,
                    "task_type": task_type,
                    "case": case,
                    # 真值子任务序列——oracle-free 条件的输入。
                    # JITPI05_ORACLE_HOLD_FIRST=1 时只留第一条：整回合只教第一步，
                    # 用来单独问「低层能不能完成第一个原子子任务」。
                    "oracle_subtasks": _oracle_subtasks(task_type, case),
                    # 分阶段进度用的符号状态对（goal.json）。LIBERO 的判据是整任务
                    # 二值的，这份标注给出每步的中间目标，见 jitpi05/progress.py。
                    "progress_goals": ground_truth_goals(task_type, case),
                    # 分阶段进度用的符号状态对（goal.json）。LIBERO 的判据是整任务
                    # 二值的，这份标注给出每步的中间目标，见 jitpi05/progress.py。
                    "progress_goals": ground_truth_goals(task_type, case),
                    # 分阶段进度用的符号状态对（goal.json）。LIBERO 的判据是整任务
                    # 二值的，这份标注给出每步的中间目标，见 jitpi05/progress.py。
                    "progress_goals": ground_truth_goals(task_type, case),
                }
            )
    return tuple(specs)


def ground_truth_subtasks(task_type: str, case: str) -> list[str]:
    """真值子任务序列（有序）——oracle 规划器条件的输入。"""
    import json

    p = bench_root() / "RoboCerebraBench" / task_type / case / "task_description.json"
    with open(p, encoding="utf-8") as fh:
        rows = json.load(fh)
    out = []
    for r in rows:
        step = str(r.get("step", ""))
        out.append(step[len("Step: "):].strip() if step.startswith("Step: ") else step)
    return out


def ground_truth_goals(task_type: str, case: str) -> dict:
    """每步的符号状态对（``goal.json``）——可机检的分阶段进度。"""
    import json

    p = bench_root() / "RoboCerebraBench" / task_type / case / "goal.json"
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def stage_progress(goals: dict, satisfied: set[str]) -> tuple[int, int]:
    """给定已满足的 ``state_pair`` 集合，返回（已完成步数, 总步数）。

    ``goal.json`` 形如 ``{"butter_1": [{"state_pair": ["Pick","butter_1"],
    "task_step": 0}, ...]}``；把每条 state_pair 序列化成 ``"|"`` 连接的字符串
    即可比较。
    """
    by_step: dict[int, list[list[str]]] = {}
    for pairs in goals.values():
        for entry in pairs:
            by_step.setdefault(int(entry["task_step"]), []).append(list(entry["state_pair"]))
    done = 0
    for step in sorted(by_step):
        if all("|".join(p) in satisfied for p in by_step[step]):
            done += 1
        else:
            break
    return done, len(by_step)
