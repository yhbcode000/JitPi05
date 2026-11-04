"""RoboCerebra 的分阶段进度探针。

为什么需要它：LIBERO 的成功判据是**整任务二值**的，而本仓库反复测到的失败形态是
「跑满步数、reward 恰好 0」——二值判据在那种地方什么也区分不了。RoboCerebra 的
``goal.json`` 提供了每一步的**符号状态对**：

    {"cream_cheese_1": [{"state_pair": ["Pick", "cream_cheese_1"], "task_step": 0},
                        {"state_pair": ["In", "cream_cheese_1",
                                        "white_storage_box_1_bottom_side"],
                         "task_step": 1}], ...}

实测确认：其中的 ``In`` 谓词名与区域名（``white_storage_box_1_bottom_side`` 等）
**与 LIBERO 自己的 ``parsed_problem["goal_state"]`` 完全一致**，并且都在 env 的
``objects_dict`` / ``fixtures_dict`` / ``object_sites_dict`` 里注册过。所以这里不自己
造几何判据，直接调 LIBERO 的 ``ObjectState.check_contain`` / ``check_ontop``。

``Pick`` 不在 LIBERO 的谓词表里（那张表是 In/On/Up/Stack/Open/Close/TurnOn…），
用「物体被抬离初始高度超过阈值」近似，并同时记录是否与夹爪接触，供事后甄别。

进度定义：按 ``task_step`` 升序，**连续**满足的状态对才算数——第一个没满足就停。
这样「完成了 3 步」意味着前 3 步都真的达成了，而不是零散地碰到过。
"""

from __future__ import annotations

from typing import Any

import numpy as np

# "Pick" 的近似的抬升阈值（米）。取 3cm：远高于仿真噪声，又低于任何一次真正抓取。
PICK_LIFT_THRESHOLD = 0.03


def resolve_control(vector_env: Any) -> Any | None:
    """从 LeRobot 的 vector env 里取到 LIBERO 任务环境（含 object_states_dict）。

    链路：SyncVectorEnv.envs[0] -> LiberoEnv ->._env(OffScreenRenderEnv) -> .env(LIBERO)
    ⚠ 底层是惰性创建的，**必须先 reset 过**，否则 ``_env`` 还是 None。
    """
    try:
        children = getattr(vector_env, "envs", None)
        if not children:
            return None
        wrapper = getattr(children[0], "_env", None)
        if wrapper is None:
            return None
        control = getattr(wrapper, "env", None)
        if control is None or not hasattr(control, "object_states_dict"):
            return None
        return control
    except Exception:
        return None


def object_positions(control: Any, names: list[str]) -> dict[str, list[float]]:
    """读取指定物体在仿真里的世界坐标。读不到的跳过，不抛。"""
    out: dict[str, list[float]] = {}
    body_id = getattr(control, "obj_body_id", {}) or {}
    sim = getattr(control, "sim", None)
    if sim is None:
        return out
    for name in names:
        idx = body_id.get(name)
        if idx is None:
            continue
        try:
            pos = np.asarray(sim.data.body_xpos[idx], dtype=float)
        except Exception:
            continue
        out[name] = [float(pos[0]), float(pos[1]), float(pos[2])]
    return out


def eef_position(control: Any) -> list[float] | None:
    """末端执行器在仿真里的世界坐标。拿不到就返回 None。

    用来区分「命令被忽略」与「命令照做了但方向不对」——只看物体抬没抬起来
    分不出这两种。
    """
    if control is None:
        return None
    for site in ("gripper0_grip_site", "gripper0_grip_site_cylinder"):
        try:
            sid = control.sim.model.site_name2id(site)
            if sid >= 0:
                p = np.asarray(control.sim.data.site_xpos[sid], dtype=float)
                return [float(p[0]), float(p[1]), float(p[2])]
        except Exception:
            continue
    return None


def _contain(control: Any, obj: str, region: str) -> bool:
    d = control.object_states_dict
    a, b = d.get(obj), d.get(region)
    if a is None or b is None:
        return False
    for fn in ("check_contain", "check_ontop", "check_contact"):
        f = getattr(a, fn, None)
        if callable(f):
            try:
                if bool(f(b)):
                    return True
            except Exception:
                continue
    return False


def satisfied_pairs(
    control: Any,
    goals: dict,
    initial_z: dict[str, float],
) -> set[str]:
    """返回当前已满足的状态对，元素是 ``"Pick|butter_1"`` 这种 ``"|"`` 连接串。"""
    done: set[str] = set()
    if control is None:
        return done
    for pairs in goals.values():
        for entry in pairs:
            pair = [str(x) for x in entry.get("state_pair", [])]
            if len(pair) < 2:
                continue
            key = "|".join(pair)
            pred, args = pair[0], pair[1:]
            if pred == "Pick":
                name = args[0]
                idx = (getattr(control, "obj_body_id", {}) or {}).get(name)
                if idx is None:
                    continue
                try:
                    z = float(control.sim.data.body_xpos[idx][2])
                except Exception:
                    continue
                if z - initial_z.get(name, z) > PICK_LIFT_THRESHOLD:
                    done.add(key)
            elif len(args) == 2:
                if _contain(control, args[0], args[1]):
                    done.add(key)
    return done


def ordered_progress(goals: dict, satisfied: set[str]) -> tuple[int, int]:
    """按 ``task_step`` 升序数**连续**完成的步数。

    同一 ``task_step`` 上的多个状态对全部满足才算这一步完成。
    """
    by_step: dict[int, list[str]] = {}
    for pairs in goals.values():
        for entry in pairs:
            pair = [str(x) for x in entry.get("state_pair", [])]
            if not pair:
                continue
            by_step.setdefault(int(entry.get("task_step", 0)), []).append("|".join(pair))
    done = 0
    for step in sorted(by_step):
        if all(k in satisfied for k in by_step[step]):
            done += 1
        else:
            break
    return done, len(by_step)


class StageProgressProbe:
    """把一个任务的 goal.json 包成可复用的探针。

    用法::

        probe = StageProgressProbe(control, task_spec["progress_goals"])
        probe.snapshot()      # -> {"stage_done": 1, "stage_total": 6, ...}
    """

    def __init__(self, control: Any, goals: dict | None, names: list[str] | None = None):
        self.control = control
        self.goals = goals or {}
        self.names = list(names or [])
        if not self.names:
            self.names = sorted(
                {str(a) for pairs in self.goals.values()
                 for e in pairs for a in e.get("state_pair", [])[1:]}
            )
        pos = object_positions(control, self.names) if control is not None else {}
        self.initial_z = {k: v[2] for k, v in pos.items()}
        self.initial_pos = dict(pos)
        # 最近一次 snapshot 量到的「连续完成步数」。oracle 用它决定何时推进。
        self.last_done = 0

    @property
    def enabled(self) -> bool:
        return bool(self.goals) and self.control is not None

    def snapshot(self) -> dict:
        if not self.enabled:
            return {}
        sat = satisfied_pairs(self.control, self.goals, self.initial_z)
        done, total = ordered_progress(self.goals, sat)
        self.last_done = done
        pos = object_positions(self.control, self.names)
        lift = {
            k: round(v[2] - self.initial_z.get(k, v[2]), 4) for k, v in pos.items()
        }
        eef = eef_position(self.control)
        return {
            "stage_done": done,
            "stage_total": total,
            "stage_satisfied": sorted(sat),
            "object_lift_m": lift,
            "eef_pos": ([round(v, 4) for v in eef] if eef else None),
        }
