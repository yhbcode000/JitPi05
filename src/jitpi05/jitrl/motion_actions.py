"""动作级（不点名物体）指令工作区。

为什么需要它——本仓库在 RoboCerebra 上实测到的一件事：

    把真值子任务「Pick up cream cheese from coffee table」按住整整 2000 步
    （oracle 条件，按达成推进，所以它从没换过），冻结的 π₀.₅ **一次都没碰
    奶油奶酪**，却把黄油抬起了 0.49–0.61 米；6 个回合里 6 次都是这样。
    而黄油正是三个目标物体里离基座最近（0.380 m）且在正前方（y≈0）的那个，
    奶油奶酪偏到 y=+0.21。

这与 SARL（arXiv:2606.31958）§5.4 报告的失效形态完全一致：

    "referencing the 'hammer' by name mistakenly causes the VLA to grasp a
     spoon, and only spatial commands (e.g., 'move right') successfully guide
     the VLA."

SARL 的做法是把语义动作从「任务级、点名物体」换成「动作级、不点名」，再由
在线 RL 学出哪句话真的能指挥得动这个底座。本模块给出那套词表的固定版本，供
两条路径使用：

  * ``motion-prog``：把一条命令序列直接当低层语言条件按固定节奏轮换——用来
    最便宜地回答「动作级指令到底能不能驱动这个底座」。
  * 后续可把它接成高层候选集（对应 ``semantic_actions`` 的角色）。

命令一律取自 SARL 正文出现过的那几种：``move down`` / ``move left`` /
``move right`` / ``close the gripper`` / ``open the gripper`` /
``move down and grasp``。**刻意不含任何物体名。**
"""

from __future__ import annotations

# 动作级指令词表。顺序即展示顺序。
MOTION_COMMANDS: tuple[str, ...] = (
    "move down",
    "move up",
    "move left",
    "move right",
    "move forward",
    "move backward",
    "close the gripper",
    "open the gripper",
    "move down and grasp",
)

# 默认程序：把「横移到目标上方 → 下去 → 夹住 → 抬起」串一遍。
# cream_cheese 相对黄油在 +y 方向偏 0.23 m，左右两个方向各备一条，见 README。
DEFAULT_PROGRAM = ("move right", "move down", "close the gripper", "move up")
MIRROR_PROGRAM = ("move left", "move down", "close the gripper", "move up")


def parse_program(raw: str) -> tuple[str, ...]:
    """把 ``"move right|move down|close the gripper"`` 解析成命令元组。

    分隔符用 ``|`` 而不是逗号：命令本身含空格，逗号容易和自然语言混淆。
    空串返回默认程序。
    """
    if not raw or not raw.strip():
        return DEFAULT_PROGRAM
    parts = tuple(x.strip() for x in raw.split("|") if x.strip())
    return parts or DEFAULT_PROGRAM


def condition_for_chunk(
    program: tuple[str, ...],
    chunk_index: int,
    chunks_per_command: int,
) -> str:
    """第 ``chunk_index`` 个低层 chunk 该用哪条命令。

    按固定节奏轮换并**循环**：程序跑完一轮从头再来，好让整回合都有指令可用。
    """
    if not program:
        return ""
    per = max(1, int(chunks_per_command))
    return program[(int(chunk_index) // per) % len(program)]
