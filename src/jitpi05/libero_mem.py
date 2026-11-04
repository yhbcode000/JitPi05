"""LIBERO-Mem（arXiv:2511.11478, AAAI 2026）suite 注册 + 有状态 ``Sequence`` 判定的接入。

为什么需要它
------------
本卷此前所有记忆实验都跑在「当前观测决定正确动作」的面板上——记忆在形式上
*不是必需的*。LIBERO-Mem 的 10 个任务用 BDDL 的 ``Sequence`` / ``Or`` 写目标，
其中六个是**同一个合取重复 N 次**：

    (:goal (Sequence (And (On akita_black_bowl_1 plate_1))
                     (And (On akita_black_bowl_1 plate_1))
                     (And (On akita_black_bowl_1 plate_1))))

状态「碗在盘子上」在**开始与结束时都成立**，中间也是真真伪伪——只有状态转移的
历史能区分你做到了第几次。这就是记忆存在的形式化前提本身。

接入要解决的两件事
------------------
1. **标准栈求不了 ``Sequence``。** LIBERO 的 ``_check_success`` 原注释写着
   "Consider conjunction goals at the moment"——只支持扁平合取；``bddl`` 与
   ``robosuite`` 里都没有 ``Sequence`` 这个词。原作者的 fork 改了两个文件
   （``bddl_base_domain.py`` 加状态与计时器、``libero_kitchen_tabletop_manipulation.py``
   加 ``_eval_goal``）。这里整块搬过来、用猴子补丁挂上，**不改 ``third_party``
   里的 libero 源码**。
2. **判定必须被推进。** 原实现把计时器放在 ``_check_success(inc=True)`` 里，而
   ``env.step`` 内部那次调用是**不推进**的探测；他们自己的采集脚本是在循环里
   手动调 ``env._check_success(inc=True)``。本模块把 ``step`` 包一层，让每次
   环境步自动推进一次，这样现有的 rollout 不用改。

⚠ 向后兼容：非 ``Sequence`` 目标走 ``is_leaf_level=True`` 那条分支，与原来的
   扁平合取行为**完全一致**；计时器代码只在 ``sequence`` 分支里跑。所以补丁对
   已有的 LIBERO-10/90/Pro 厨房任务是零影响（已实测）。

⚠ init 文件的来源：原仓库只预生成 4 个任务的 ``pruned_init``。已核对 ``:init``
   谓词——任务 1–6 **完全同构**、任务 9 与 10 同构，所以用它们的 init 文件互相
   复用（初始状态本来就是采样出来的合法摆放，目标不同不影响其有效性）。
   这顺带给出一个好性质：**任务 1/2/3/4/5/6 共享同一批初始状态**，
   是「同一初始状态、不同目标」的直接配对。
"""

from __future__ import annotations

import glob
import os

from libero.libero import benchmark as _benchmark
from libero.libero.benchmark import (
    Benchmark,
    Task,
    grab_language_from_filename,
    register_benchmark,
)
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

SUITE = "libero_mem"

# 判定器里要求「真 >5 步」且「假 >15 步」才记一次子目标达成——这是原作者用来
# 滤掉瞬时抖动的阈值，也是「举起再放下算一次」能成立的原因。原样保留。
LIVE_THRESHOLD = 5
NONLIVE_THRESHOLD = 15

_installed = False


# ====================================================================== 判定器
def _lm_reset_subgoal_progress(self) -> None:
    self._satisfied_subgoals = []
    self._sub_goal_live_time = 0
    self._sub_goal_nonlive_time = 0


def _lm_is_subgoal_trajectory_possible(self, current_traj, testing_traj) -> bool:
    """当前已达成的子目标序列是否是某条 ``Or`` 分支的前缀。"""
    set_a = {tuple(item) for clause in current_traj for item in clause[1:]}
    set_b = {tuple(item) for clause in testing_traj for item in clause[1:]}
    return set_a.issubset(set_b)


def _lm_eval_goal(self, goal_state, inc=False, is_leaf_level=False):
    """递归求值：支持 ``and`` / ``or`` / ``sequence`` / 原子谓词。

    原实现逐字搬自他们的
    ``libero/libero/envs/problems/libero_kitchen_tabletop_manipulation.py``。
    """
    if is_leaf_level:
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result

    op = goal_state[0]
    if op == "and":
        return bool(self._eval_goal(goal_state[1:], inc, is_leaf_level=True))

    if op == "sequence":
        idx = len(self._satisfied_subgoals)
        if self._overshot:  # 一旦过冲，此后永远判负
            return False
        if idx >= len(goal_state) - 1:
            idx -= 1
        subgoal = goal_state[idx + 1]  # 偏移 1：goal[0] == 'sequence'
        assert len(subgoal[1:]) == 1, "sequence 的每一步只含一个 'and' 子句"
        success = self._eval_goal(subgoal, inc, is_leaf_level=is_leaf_level)

        if success:
            if len(self._satisfied_subgoals) >= len(goal_state) - 1:
                self._sub_goal_nonlive_time = 0
                return True
            if (
                self._sub_goal_live_time > LIVE_THRESHOLD
                and self._sub_goal_nonlive_time > NONLIVE_THRESHOLD
            ):
                self._sub_goal_live_time = 0
                self._sub_goal_nonlive_time = 0
                self._satisfied_subgoals.append(subgoal)
                return True
            if inc:
                self._sub_goal_live_time += 1
        else:
            if len(self._satisfied_subgoals) >= len(goal_state) - 1:
                if self._sub_goal_nonlive_time > 5:
                    self._overshot = True
                    return False
            if inc:
                self._sub_goal_nonlive_time += 1
        return False

    if op == "or":
        for subgoal in goal_state[1:]:
            assert subgoal[0] == "sequence"
            if not self._is_subgoal_trajectory_possible(
                current_traj=self._satisfied_subgoals, testing_traj=subgoal[1:]
            ):
                continue
            if self._eval_goal(subgoal, inc, is_leaf_level=is_leaf_level):
                return True
        return False

    raise ValueError(f"未知的目标结构：{goal_state}")


def _lm_check_success(self, inc: bool = False):
    """有状态的成败判定。``inc=True`` 时推进计时器（每个环境步推进一次）。"""
    goal_state = self.parsed_problem["goal_state"]
    first = goal_state[0][0]
    is_leaf_level = first != "sequence" and first != "or"
    if not is_leaf_level:
        goal_state = goal_state[0]
    return bool(self._eval_goal(goal_state, inc, is_leaf_level=is_leaf_level))


def _lm_progress(self) -> dict:
    """给记录用的进度读数：已达成几步、共几步、是否过冲。"""
    goal_state = self.parsed_problem["goal_state"]
    total, kind = 0, "flat"
    first = goal_state[0][0]
    if first == "sequence":
        kind, total = "sequence", len(goal_state[0]) - 1
    elif first == "or":
        kind, total = "or", max(len(b) - 1 for b in goal_state[0][1:])
    return {
        "goal_kind": kind,
        "satisfied": len(getattr(self, "_satisfied_subgoals", [])),
        "total": total,
        "overshot": bool(getattr(self, "_overshot", False)),
    }


def install() -> None:
    """把有状态判定器挂到 libero 的厨房任务类上（幂等）。"""
    global _installed
    if _installed:
        return

    # ⚠ 不能按模块属性取——LIBERO 用 ``@register_problem`` 装饰器注册，而那个装饰器
    # 返回 None，于是 ``模块.Libero_Kitchen_Tabletop_Manipulation`` 就是 None。
    # 必须从 TASK_MAPPING 里拿真正的类对象（键是类名的小写）。
    from libero.libero.envs import TASK_MAPPING

    _Kitchen = TASK_MAPPING["libero_kitchen_tabletop_manipulation"]
    _orig_init = _Kitchen.__init__
    _orig_step = _Kitchen.step

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._satisfied_subgoals = []
        self._sub_goal_live_time = 0
        self._sub_goal_nonlive_time = 0
        self._overshot = False

    def _step(self, action):
        # ⚠ 原 step 内部那次 _check_success() 是**不推进**的探测。真正推进计时器的
        # 是这里这一次。不这样做，Sequence 永远不会前进（这是接入的头号坑）。
        obs, reward, done, info = _orig_step(self, action)
        done = self._check_success(inc=True)
        return obs, reward, done, info

    _Kitchen.__init__ = _init
    _Kitchen.step = _step
    _Kitchen._check_success = _lm_check_success
    _Kitchen._eval_goal = _lm_eval_goal
    _Kitchen._is_subgoal_trajectory_possible = _lm_is_subgoal_trajectory_possible
    _Kitchen.reset_subgoal_progress = _lm_reset_subgoal_progress
    _Kitchen.libero_mem_progress = _lm_progress

    _installed = True


# ====================================================================== suite
def _bddl_dir() -> str:
    """定位已铺好的 bddl 目录（安装步骤脚本会把文件拷到这里）。"""
    import libero.libero as _pkg

    return os.path.join(os.path.dirname(_pkg.__file__), "bddl_files", SUITE)


def _scene_no(name: str) -> int:
    """从 `KITCHEN_SCENE1_<n>_...` 里取场景号。⚠ 不能按字符串排序——那样 10 会排到 1 前面。"""
    try:
        return int(name.split("_")[2])
    except (IndexError, ValueError):
        return 10**6


def task_names() -> list[str]:
    """按**场景号**排序，顺序稳定（task_id 依此确定）。"""
    names = sorted(
        (os.path.basename(p)[:-5] for p in glob.glob(os.path.join(_bddl_dir(), "*.bddl"))),
        key=_scene_no,
    )
    if not names:
        raise RuntimeError(
            "找不到 LIBERO-Mem 的 bddl 文件。先跑 install_libero_mem_assets.sh 把 "
            f"{SUITE} 的 bddl/init 铺进 libero 包目录。"
        )
    return names


def register_libero_mem_suite() -> None:
    """把 ``libero_mem`` 注册进 libero benchmark（幂等）。"""
    install()
    if SUITE in _benchmark.BENCHMARK_MAPPING:
        return

    names = task_names()
    libero_task_map[SUITE] = list(names)
    _benchmark.task_maps[SUITE] = {}
    for name in names:
        _benchmark.task_maps[SUITE][name] = Task(
            name=name,
            language=grab_language_from_filename(name + ".bddl"),
            problem="Libero",
            problem_folder=SUITE,
            bddl_file=f"{name}.bddl",
            init_states_file=f"{name}.pruned_init",
        )

    def __init__(self, task_order_index: int = 0) -> None:
        Benchmark.__init__(self, task_order_index=task_order_index)
        self.name = SUITE
        self._make_benchmark()

    register_benchmark(type("LIBERO_MEM", (Benchmark,), {"__init__": __init__}))


def libero_mem_task_specs(max_steps: int | None = None) -> tuple[dict, ...]:
    """产出与其它面板同构的任务 spec 列表。"""
    register_libero_mem_suite()
    bench = _benchmark.get_benchmark_dict()[SUITE]()
    out = []
    for tid in range(bench.n_tasks):
        task = bench.get_task(tid)
        out.append(
            {
                "name": f"libero_mem_task{tid}",
                "suite": SUITE,
                "task_id": tid,
                "description": task.language,
                "max_steps": max_steps if max_steps is not None else 520,
                "zero_shot": True,
            }
        )
    return tuple(out)
