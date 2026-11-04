# MANIFEST · PrefMem-JiT vs LoRA-acc（fail-closed · runnable）

**STATUS: READY**（自 2026-11-01T09:21:31+08:00：B 臂 8/8 mission 跑通；此前为 BLOCKED@B9）
**STATUS=READY 依据**：B 臂执行链 **8/8 mission 实测跑通**（`done: 8 ok, 0 failed`，8 个 `rollout_summary.yaml`，`artifacts/hf_pull_logs/b9_run8_summaries/`）；运行时规格 `py3.11 + robosuite 1.4.1 + mujoco 2.3.7 + numpy 1.26.4`；唯一 infra patch = benchmark 的 bddl 缺失容错。**正式开跑前仍需**：FC2（脏 PDF 认定）、`seed=0`（dry-run 用的是 runfile 默认 7）、每 task 5 episodes、FC13（A 臂历史基线有效性确认）。

**B selector v2（2026-11-01T10:3x）**：`local_shims/jit_selector_4b.py` 用**真实 Qwen3.5-4B**（4,539,265,536 参数，CPU bf16）首 token logits 取代均匀 pi0；8/8 mission 非均匀（std 0.125–3.0）。已知缺陷：仓内无 Qwen 侧 LoRA 适配器（299 grounding 在执行器侧）、冷启动 `Â≡0`、本 dry-run 为 selector-only（GPU 被 A 训练占满）。
- `hypothesis_id_readonly`: `PrefMem-JiT-vs-LoRA-acc (Haobo DIRECT 2026-11-01)`
- 本文件性质：只读关联 + 可复现 runnable manifest。**不构成开训授权**（授权见 §12 gate）。
- 硬边界（本轮实际写入/未写入）：
  - **已写**：本 MANIFEST（4 份副本）；`prefmem-libero/local_shims/`（4 个附加文件：注册 shim / 运行 shim / SERVER_PY wrapper / README）；WSL ext4 环境 `~/projects/{openpi,prefmem-libero,ckpt/299}` 与两个 venv。
  - **未写**：训练代码（vendor 与 prefmem 脚本均未改；`scripts/` 277 文件基线 sha `5d1a35df…` 不变）；原始 ckpt 299（只读复制）；数据/权重原件；任何 `full74*` / `.109 full74v3` / Dig 产物；未 push；**未开训**。
- 状态唯一权威：本文件 `STATUS` 行。任何 `READY` 必须有 §11 逐条 `RESOLVED`（带证据路径/MD5/时间戳）支撑，禁止口头清零。

```yaml
# 机器可读摘要（供 gate/脚本解析；与正文冲突时以正文为准）
manifest_id: jitrl_vs_lora_acc
status: READY
status_basis: "fail-closed: 静态检查两侧 PASS（init-state bundle 245e6f4d… 16/16 一致；D1 CLOSED）；**A/B 数值在 D2 解决前不可比**（任务 3-2_032_7 指令文本 ALWM≠4090，见 §6.12 D2）；A 臂 eval ckpt = PENDING Lead；B 在 4090 = PROVISIONAL，ALWM 一致性 = DEFERRED"
hypothesis_id_readonly: "PrefMem-JiT-vs-LoRA-acc (Haobo DIRECT 2026-11-01)"
repo: "https://github.com/LifeCheckpoint/JitPi05.git"
branch: jitrl-vs-lora-acc
commit: "5472f1f43e84750cb2b0760b17b5853cd788eddb"
assets_ckpt_dir: "checkpoints/pi05_libero_lora_neutral_v5init/pi05_neutral_lora_v4_v5init/299"
assets_ckpt_files: 28
assets_ckpt_bytes: 9363286152
assets_ckpt_md5_list_sha256: "977eb6e9d52630877c938206ddf883f6b4e3aaa635ea7a7b286155355c219cc2"
assets_ckpt_size_manifest_sha256: "a9ee6806bb48b56662c7fb3c8255d1f974f9624336bd5cb6a4888a6b63819ca5"
assets_ckpt_parity: "byte-identical on ALWM18R2 and 2026-552A-4090"
arm_a_entry: "prefmem-libero/scripts/qwen_lora_rl.py"
arm_a_run_form: "OPENPI_PY=$PREFMEM_ROOT/local_shims/serve_py_wrapper.sh + python ../local_shims/run_local_a_arm.py"
arm_b_entry: "prefmem-libero/scripts/run_plan_with_vla.py (+ plan_jitrl_* selector; hop code exists, server-half proven, env-half blocked)"
arm_b_blocker: "B9 (env builds; protocol seam KeyError observation/image at libero_policy.py:52; manual 4/5-key probes OK)"
motive_cn: "前置判断「JitRL 在 VLA 只能做偏好训练」不可靠——先前实验把 Jit 放在了错误的槽位。已知：偏好训练有效。未知：JitRL 能否完全替代 VLA RL。本线在以 grounded LoRA 微调为基线的场景上重做对照：A=grounded LoRA RL；B=测试期 JitRL 等价替换（z'=z+β·A；post-LoRA 的 JitPi 偏好对齐参考实现）。目标=证明可替换性，而非仅偏好训练有效。"
motive_en: "Prior judgment 'JitRL in VLA can only do preference training' is UNRELIABLE because the earlier experiment put Jit in the wrong slot. Known: preference training works. Unknown: whether JitRL can FULLY REPLACE VLA RL. This line retrains/contrasts on the grounded-LoRA-finetune case: A = grounded LoRA RL; B = test-time JitRL equivalent replace (z'=z+β·A; post-LoRA JitPi preference-alignment reference). Goal = prove replaceability, not merely preference-training efficacy."
contrast_narrative_cn: "论文主张 JitRL 可在不更新权重的前提下替换 RL；本实验的对照不是换任务套件，而是在同一 grounded LoRA 训练设定内替换更新规则：A=接地 LoRA RL（Qwen3.5-4B 上 qwen_lora_rl 权重更新 + 冻结执行器 299）；B=接地 LoRA 之后的 JitPi 偏好对齐（测试期 z'=z+β·A，无新增权重训练入口）。评测=同 8 个 prefmem 任务 / 同 backbone / 同 299 上的任务执行准确率 A vs B。"
contrast_narrative_en: "The JitRL paper claims it can replace RL without weight updates. This line's contrast is NOT a different task suite but a swap of the update rule inside the same grounded-LoRA setup: A = grounded LoRA RL (weight updates on Qwen3.5-4B via qwen_lora_rl + frozen executor 299); B = post-grounded-LoRA JitPi preference alignment (test-time z'=z+β·A; no new weight-train entry). Eval = task accuracy A vs B on the same 8 prefmem tasks / same backbone / same 299."
hop_probe: "local_shims/jit_hop_probe.py — fake decision -> ws 299 -> actions [10,7], ok=true, 2026-11-01T07:52:38+08:00"
b_selector_v2: "local_shims/jit_selector_4b.py — real Qwen3.5-4B first-token logits (CPU bf16), 2026-11-01T10:3x"
b_selector_model: "/models/Qwen3.5-4B (4,539,265,536 params; shards 8e5bd999.../011b980f...)"
b_selector_deficits: "base-4B pi0 (no Qwen LoRA adapter in tree); cold start -> A-hat==0; CPU-only because A-run owns GPU"
b_selector_venv: "~/projects/jitrl_sel4b/.venv (py3.11; torch 2.15.0+cpu; tf 5.5.4; 39 pkgs; freeze sha256 58d369f98e053b29fef7fcdf13209ea31563c10342e45bfd04f0d433a742ee46)"
shared_venv_frozen: "~/projects/libero_rt311/.venv = shared A/B rollout client, Lead-frozen, 95 pkgs, freeze sha256 d9efcd2b5c63ead92e130c7d3ba62d74a21b729189b57f72e323928fd8c3a7fe"
eval_protocol: "task accuracy (A: sr/strict/compliance fields) on the same 8 prefmem mission ids, seed 0, 12 train / 5 eval episodes per task; B uses test-time selection (no weights produced)"
ruling_2026_11_01_0744: "B = JitRL test-time equivalent replace; weight-train demand withdrawn"
ruling_2026_11_01_0745: "B must name the JitPi preference-alignment path over grounded LoRA"
superseded: ["2026-11-01T07:36 weight-train mandate (B8)"]
ruling_option: "TRUE_ALIGNMENT_REPLACE"
ruling_date: "2026-11-01T~07:36+08:00"
ruling_source: "Haobo via Lead Explore"
aligned_tasks: [3-1_004_1_blue, 3-1_006_1, 3-2_032_7, 3-2_032_8, 3-2_097_8, 3-2_097_10, 3-9_118_1, 3-10_117_1]
shared_backbone: "Qwen3.5-4B (models/Qwen3.5-4B; 43 files; 9342908952 B)"
shared_executor: "pi05_neutral_lora_v4_v5init/299 (md5-list 977eb6e9...)"
host_alwm18r2:
  root: "E:/Envs/JitRL_LoRA_Env"
  gpu: "RTX 4090 Laptop 16376 MiB; used 1318 MiB; 2%; 52C; 2026-11-01T01:15:25+08:00"
  venv: absent
  wsl_ubuntu: present
host_pc4090:
  root: "D:/Envs/JitRL_LoRA_Env"
  gpu: "RTX 4090 24564 MiB; used 12588 MiB; 0%; 43C; 2026-11-01T01:15:25+08:00"
  venv: absent
  wsl_ubuntu: absent
  ssh_alias: pc4090
  hostname: "2026-552A-4090"
blockers: [D2]
blockers_resolved: [B1, B1b, B2, B3, B4, B5, B6, B7, B8, B9, B10]
blocker_notes: "全部清零：B9 以 8/8 mission rollout_summary + done: 8 ok 关闭；B9c（suite 注册 40→45）已随 B9 关闭"
fail_closed_assertions: "15 rows (FC1-FC15; FC3 split a/b)"
fail_closed_assertions_passing: [FC1, FC3a, FC3b, FC4, FC5]
blockers_semantics: "YAML `blockers` 列表 = 跨臂可比性 gate（当前 [D2]）；§11 B-list blocker 状态另见 FC1 行，二者不混淆"
gate: LEAD_EXPLORE_START_ORDER
output_dir_template: "<root>/artifacts/jitrl_vs_lora_acc/<run_id>/{arm_a,arm_b}"
run_id_template: "20261101T<HHMM>_seed0_5472f1f"
```

---

## 1. 科学问（Haobo 裁定 2026-11-01，两段：07:36 → **07:44 覆盖** → 07:45 addendum）


### 1.0 科学动机（钉死，2026-11-01T~07:54+08:00；CN + EN）

- **CN**：前置判断「JitRL 在 VLA 只能做偏好训练」**不可靠** —— 因为先前实验把 Jit 放在了**错误的槽位**。
  **已知**：偏好训练（preference training）有效；**未知**：JitRL 能否**完全替代** VLA RL。
  本线在以 **grounded LoRA 微调**为基线的场景上重做对照：**A = grounded LoRA RL**；**B = 测试期 JitRL 等价替换**
  （`z'=z+β·A`；即 post-LoRA 的 JitPi 偏好对齐参考实现）。**目标 = 证明"可替换性"（replaceability），而不仅是偏好训练有效**。
- **EN**：The prior judgment *"JitRL in VLA can only do preference training"* is **UNRELIABLE** because the earlier
  experiment put Jit in the **wrong slot**. **Known**: preference training works. **Unknown**: whether JitRL can
  **fully replace** VLA RL. This line re-runs the contrast on the **grounded-LoRA-finetune** case:
  **A = grounded LoRA RL**; **B = test-time JitRL equivalent replace** (`z'=z+β·A`, i.e. the post-LoRA JitPi
  preference-alignment reference). **Goal = prove replaceability**, not merely preference-training efficacy.

**最终有效口径（07:44）**：B 臂 = **JitRL 测试期路径**（`z' = z + β·A`，closed-form logit/score 重加权），作为 A 臂 **LoRA RL 训练槽位的等价替换**；两臂必须同 backbone（Qwen3.5-4B）、同 8 个 prefmem 任务、同执行器（`pi05_neutral_lora_v4_v5init/299` 线）；度量 = 任务执行准确率。

**07:45 addendum**：B 臂必须点名并使用「接地 LoRA 加载后的 **JitPi 偏好对齐**」参考路径。

| 臂 | 定义 | 实现 | 状态 |
|---|---|---|---|
| A（基线） | grounded LoRA + LoRA RL（权重更新） | `prefmem-libero/scripts/qwen_lora_rl.py`（`loss.backward()`/`opt.step()`，`:233-237`） | 环境已验（B1b） |
| B（处理） | JitRL 测试期：`π_β ∝ π_0·exp(β·A)`，`A=Q−V`，无梯度 | `prefmem-libero/scripts/plan_jitrl_*` + `controlled_loop_109.py`（参考路径已定位，见 §6.2） | **不可运行**（B9/B10） |

**审计痕迹（不可删除）**：
- 07:36 版裁定（"B 必须权重训练、禁止 test-time"）已被 07:44 覆盖 → **superseded**；据此开立的原 B8（"权重训练无定义"）一并关闭（§11 B8）。
- 仍**永久禁止**：用 `Qwen3.5-2B` 充当对照（backbone 不匹配）；把 Jit **叠加**在 LoRA 之上。
- 论文口径注记：`JITRL.pdf` 为 gradient-free 定义（标题/摘要/§4.3 `z'=z+β·Â`）；`prefmem-libero/docs/preference_plan_story.typ:325` 记录**主结果用 argmax-Q**、`β·A` 属原型形式 —— 两者都属"测试期"，与 07:44 口径一致；**hop 代码存在**（§6.5 `run_plan_with_vla.py`），**环境半未验**（B9b）。

---

## 2. 仓 / 分支 / commit（双机已核验）

| 项 | 值 | 核验主机 |
|---|---|---|
| JitPi05 remote | `https://github.com/LifeCheckpoint/JitPi05.git`（origin，fetch/push 同名） | ALWM18R2 + 2026-552A-4090 |
| worktree 分支 | `jitrl-vs-lora-acc` | 两机一致 |
| worktree commit | `5472f1f43e84750cb2b0760b17b5853cd788eddb`（2026-09-12 11:40:01 +0800，`Add counterfactual recovery utilities and related tests`） | 两机一致 |
| 主仓 `JitPi05/` 分支 | `eliminate` @ 同 commit `5472f1f`（ALWM 同一值；4090 侧另有 26 条未提交改动，属主仓既有状态，与本 runnable 无关） | 两机核验 |
| worktree 脏状态 | ` M docs/papers/CycleVLA.pdf`，13,407,989 B，MD5 `96dc4ba22b674a74d12b2a3fa0672a15`（两机**同字节**、同为 `M`，非本轮引入） | ALWM + 4090 实测；见 FC2 |

---

## 3. 目录与环境路径（双机）

| 项 | ALWM18R2（本机，`E:`） | 2026-552A-4090（`D:`） |
|---|---|---|
| 根 | `E:\Envs\JitRL_LoRA_Env` | `D:\Envs\JitRL_LoRA_Env` |
| JitPi05 主仓 | `E:\Envs\JitRL_LoRA_Env\JitPi05` | `D:\Envs\JitRL_LoRA_Env\JitPi05` |
| JitPi05 worktree | `E:\Envs\JitRL_LoRA_Env\JitPi05-jitrl-vs-lora-acc` | `D:\Envs\JitRL_LoRA_Env\JitPi05-jitrl-vs-lora-acc` |
| prefmem-libero | `E:\Envs\JitRL_LoRA_Env\prefmem-libero` | `D:\Envs\JitRL_LoRA_Env\prefmem-libero` |
| 基线 ckpt（299） | `...\checkpoints\pi05_libero_lora_neutral_v5init\pi05_neutral_lora_v4_v5init\299\` | `...\checkpoints\...\299\`（同字节） |
| 基座 `Qwen3.5-4B`（Admin RO-copy） | `E:\Envs\JitRL_LoRA_Env\models\Qwen3.5-4B`（43 文件 / 9,342,908,952 B） | `D:\Envs\JitRL_LoRA_Env\models\Qwen3.5-4B`（同规模） |
| A 臂数据/候选 | `E:\Envs\JitRL_LoRA_Env\prefmem_lora`（files=374 / 6,269,404,610 B；其中 data 350/6,269,358,674 + openpi_assets 24/45,936）+ `prefmem-libero\runs\online_candidates{,_qwen35}` | `D:\Envs\JitRL_LoRA_Env\prefmem_lora`：**files=374 / 6,269,404,610 B（与 ALWM 同规模，实测）**；candidates 目录未核验 |
| openpi_assets | `...\openpi_assets\pi05_libero_lora_neutral_v5init` | 同布局（仅 1 文件，见 §4.3） |
| JitPi05 `.venv` | Windows：**无**（`uv sync` 失败，见 B1）；**WSL：有**（`~/projects/JitPi05-jitrl-vs-lora-acc/.venv`，6.9 GB，**230 包**（含 msgpack；freeze sha256 `88cb7c03…`），doctor EXIT=0） | **无**（出列，B5） |
| `third_party/openpi` | clone 完成 @ `15a9616a00943ada6c20a0f158e3adb39df2ccac`（gitlink 指定；27 MB；`.venv` 未建） | 空（未 clone） |
| `third_party/libero-pro` | clone 完成 @ `eafdb809426b13153aa1e4c42d6601844217dfec`（与 gitlink 一致；990 MB；环境装配未做） | 空（未 clone） |
| 根 `.venv` | **无** | **无** |
| `uv` | `C:\Users\disco\.local\bin\uv.exe`（可直接执行） | `C:\Users\disco\AppData\Local\Microsoft\WinGet\Links\uv.exe`（**符号链接，非交互 ssh 下执行报"系统无法执行指定的程序"**；须用真实路径 `...\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe`，版本 `0.13.5`） |
| `python`（Windows） | 未在 PATH 解析（`which python` 空）→ **UNKNOWN** | `C:\Program Files\Python312\python.exe`（另有 WindowsApps 存根） |
| WSL2 | `Ubuntu-24.04`（Running，v2）+ `docker-desktop` | **仅** `docker-desktop`（无 Ubuntu 发行版） |
| WSL 内 checkout（ALWM） | `~/projects/JitPi05-jitrl-vs-lora-acc` @ `5472f1f`（ext4，`/dev/sdf` 940 GB free）；`uv 0.13.6`；`python3.12`；`nvidia-smi -L` 可见 RTX 4090 Laptop | — |
| WSL 内 A 臂 env（ALWM） | `~/projects/openpi` @ `15a9616a`（`.venv` py3.11 / 7.7 GB / 242 包 / freeze `55b1afcb…`）；`~/projects/prefmem-libero`（含 `upstream/LIBERO` @ `8f1084e3…`，无仓内 pin）；`~/projects/ckpt/299`（28 文件 / 9,363,286,152 B / MD5 清单 sha `977eb6e9…`） | — |
| WSL 内 HF 缓存（ALWM） | `~/.cache/huggingface/hub`：`pi05-libero` 9.9 GB（`9ba5431256f4`）、`Qwen3.5-2B` 4.5 GB（`15852e8c1636`）、`paligemma` 22 MB（`ee1ed4a8fbec`）、`pi05_libero_base` 14 GB（`5fcf06fc42b6…`），4/4 `EXIT=0` | — |
| 磁盘余量 | `E:` 107.1 GB free；`C:` 236 GB free | `D:` 1326.0 GB free（已用 203.2 GB） |

`prefmem-libero` **无任何依赖声明**（无 `pyproject.toml` / `requirements*.txt` / `uv.lock` / `setup.py`）→ A 臂环境无法从仓内重建（见 B1）。

---

## 4. 基座 checkpoint 指纹（`299` · 双机字节一致，实测）

### 4.1 汇总指纹

| 指标 | 值 |
|---|---|
| 路径 | `E:\Envs\JitRL_LoRA_Env\checkpoints\pi05_libero_lora_neutral_v5init\pi05_neutral_lora_v4_v5init\299` |
| 文件数 | **28** |
| 总字节 | **9,363,286,152**（8.72 GiB） |
| MD5 清单 SHA-256 | `977eb6e9d52630877c938206ddf883f6b4e3aaa635ea7a7b286155355c219cc2` |
| 目录 size-manifest SHA-256 | `a9ee6806bb48b56662c7fb3c8255d1f974f9624336bd5cb6a4888a6b63819ca5` |
| 双机一致性 | **ALWM18R2 与 2026-552A-4090 的 28 个文件 MD5 逐一相等**（4090 侧用 PowerShell `Get-FileHash -Algorithm MD5` 独立计算） |

### 4.2 文件级 MD5（全 28 项；`<md5>  ./<relpath>`，按路径排序）

```text
2b2fa0ee0df9e79c5a696b581d583aca  ./_CHECKPOINT_METADATA
03d102a77c0aa343afbf7569c97bbcba  ./assets/local/prefmem_neutral/norm_stats.json
e0e33ba9aadecb59c03a2ace50d2c3d1  ./params/_METADATA
4f7b8a50540bdf1308b10dde2c4efcec  ./params/_sharding
c4f91a7205f12df2f0fdf28197d4f783  ./params/array_metadatas/process_0
9d6702191af430441950226bbc260af2  ./params/d/5c04cda6f3d3319159a4c6ea9ba3f9f4
789ec95151434aec58f94cca9757c75a  ./params/manifest.ocdbt
5023d9d6ea567c9914430daa31b85ab8  ./params/ocdbt.process_0/d/09753908ddf46d948f6eafbe21721308
08afe670ad3f162492d33c41adf0fb0c  ./params/ocdbt.process_0/d/61fdf4da3d5593dd594eafd31d973391
0a03042836a1808cf3f0719e0e8bf9ae  ./params/ocdbt.process_0/d/633690999b82848a0bb2dca4b8441e45
29892c18e2fd70337dd7fe7a38f869ed  ./params/ocdbt.process_0/d/85071fe0540712a583ed7ad74f2063a4
3225442cc371ddf641aecf4b93dee888  ./params/ocdbt.process_0/d/da1a725931f0a9dc6682dcceb9a0de87
7d222533782504533da36f23785f791e  ./params/ocdbt.process_0/d/e4a0ac84b6211683abed6bc57abc1fe4
e81d284f358d3720dd79a2e3e901d91f  ./params/ocdbt.process_0/d/e5a90068bf6e1bf7a04059ba37853719
1fc6870b3a644a872bb2c1c19f0b7073  ./params/ocdbt.process_0/d/f4f447759462bc27beb1df611f0562a7
eea04719c195359adde977cc900b125b  ./params/ocdbt.process_0/d/f65fb4f15026a9c6aa16de348d808bf6
f542e8de5ee24deeee1b432bf29ec1b6  ./params/ocdbt.process_0/manifest.ocdbt
6646ab8b0ef24400b0c79c9e598f6372  ./train_state/_METADATA
ba32a96b4d598d02dc5e7be67f4d6c0d  ./train_state/_sharding
9d9b27ece1d573dcbe7d23e3851204e2  ./train_state/array_metadatas/process_0
1481635ad591597b56a77f13d7bca6e3  ./train_state/d/0a04aa88a0daf11f0b2a30630a0f6514
3fc809ffd57b4f9692fad9a886175b0a  ./train_state/manifest.ocdbt
8ddf84514d6586f78263f6470861f06b  ./train_state/ocdbt.process_0/d/00ece0488ecd1115f9ef57da1409d0d0
f6f60eb2064bef6595b802fb1d3483b0  ./train_state/ocdbt.process_0/d/1c9fa2903fc4072aa261c2f2c97479ca
553dd2318868c9f24aa8f4021de81d2b  ./train_state/ocdbt.process_0/d/209b45133f6586ad3532eca2eedda21e
9b0a3e1a075293fa1fb092adf5ded68f  ./train_state/ocdbt.process_0/d/5af50ad241ab052d818e55484473b15b
1c215d13fff1cae2a29e851ae83ebf9a  ./train_state/ocdbt.process_0/d/849fc1209d7f9356d3a7c414ed33df86
07fabc487d1648bbfb9f5d2de82e36ad  ./train_state/ocdbt.process_0/manifest.ocdbt
```

### 4.3 邻接资产

| 路径 | 字节 | MD5 |
|---|---:|---|
| `...\299\assets\local\prefmem_neutral\norm_stats.json` | 1,914 | `03d102a77c0aa343afbf7569c97bbcba` |
| `...\openpi_assets\pi05_libero_lora_neutral_v5init\local\prefmem_neutral\norm_stats.json` | 1,914 | `03d102a77c0aa343afbf7569c97bbcba`（与 299 内副本同 MD5；**ALWM 与 4090 实测一致**） |

### 4.4 `prefmem-libero/scripts` 双机字节一致（A 臂代码基线）

| 指标 | 值 |
|---|---|
| 文件数（`-maxdepth 1 -type f`） | **277** |
| 清单 SHA-256（`<md5>  scripts/<name>`，按路径排序，LF） | `5d1a35df0320983081b89c6e1a3e9d61e397fb22d481b26e3b058eede017cc98` |
| 双机一致性 | **ALWM18R2 与 2026-552A-4090 逐文件 MD5 全等**（4090 侧 PowerShell `Get-FileHash`，输出经 CRLF→LF 归一后比对） |
| 例外 | 4090 侧多一个目录 `scripts\__pycache__`（非 `-type f`，不计入） |

> 结论：A 臂"代码同源"成立；不对等的是**运行环境**（B1）、**数据**（B2/B7）、**权重量**（B3）。

### 4.5 该 ckpt 的角色（读自仓内文档，非推断）

- `prefmem-libero/results/benchmark_protocol.md`：neutral VLA `pi05_neutral_lora_v4_v5init/299` 是 A 臂的**冻结执行器**（low-level），A 的 RL/LoRA 训练发生在高层 Qwen3.5-4B 规划策略上（`qwen_lora_rl.py` docstring: "weight-level traditional-RL counterpart to JiTRL"）。
- 该 ckpt 是 **OpenPI/OCDBT 格式**，需 `scripts/serve_pi05.py`（openpi venv）提供服务；JitPi05 侧低层走 LeRobot `lerobot/pi05-libero`。→ 见 B6b。
- 299 → OpenPI 服务的**实际调用形态**（`prefmem-libero/scripts/qwen_lora_rl.py:50-57`，`start_server()`，非推断）：

```python
env["PYTHONPATH"] = OPENPI_SRC                     # /hhd2/.../openpi/src
env["CUDA_VISIBLE_DEVICES"] = str(gpu)
env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
subprocess.Popen(
    [SERVER_PY, "scripts/serve_pi05.py",
     "--config", task["config"],
     "--checkpoint", task["checkpoint"],           # 指向 .../pi05_neutral_lora_v4_v5init/299
     "--port", str(port)],
    cwd=str(ROOT), env=env, ...)
# 随后 wait_port(port)（超时 240 s）再 sleep 15 s 才发请求
```

---

## 5. A 臂入口（LoRA FT / LoRA RL）

### 5.1 主入口（权重级 LoRA RL，科学问的"当前 LoRA RL 微调路径"）

- 脚本（相对根）：`prefmem-libero/scripts/qwen_lora_rl.py`（12,521 B）
- 运行目录：`prefmem-libero/scripts`（脚本 `from controlled_loop_109 import ...`，同目录导入）
- 真实 CLI（读自 `qwen_lora_rl.py:104-120` 的 argparse 默认值，未改写）：

```bash
# cwd = prefmem-libero/scripts
# --model 取值（本机绝对路径；同一份权重，43 文件 / 9,342,908,952 B）：
#   ALWM : E:\Envs\JitRL_LoRA_Env\models\Qwen3.5-4B
#   4090 : D:\Envs\JitRL_LoRA_Env\models\Qwen3.5-4B
#   服务器原值 : /hhd3/yanghaobo/models/Qwen3.5-4B
python3 qwen_lora_rl.py \
  --manifest <A 臂任务 manifest> \
  --task-ids <8 个 task id> \
  --candidates-root runs/online_candidates_qwen35 \
  --model "${PREFMEM_MODEL_DIR}/Qwen3.5-4B" \
  --out-dir runs/qwen35_lora_rl \
  --train-episodes 12 --eval-episodes 5 \
  --temperature 2.0 --epsilon 0.3 \
  --reward-gate sr \
  --lr 1e-5 --lora-r 16 --lora-alpha 32 \
  --exec-gpu 0 --port 8401 --seed 0 --prompt-mode legacy
# PREFMEM_MODEL_DIR 由执行者按主机导出（WSL 内为 /mnt/e/... 或拷贝到 ext4）
```

- **本地执行形态（B1b 已装配，2026-11-01）**：服务器版命令在本机不可直接用（`SERVER_PY`/`OPENPI_SRC`/`CLIENT_SRC` 硬编码 `/hhd2/...`）。改用附加 shim，argv 与上面完全一致：

```bash
export OPENPI_ROOT=~/projects/openpi                 # 15a9616a, py3.11 venv (uv sync --frozen, EXIT=0)
export PREFMEM_ROOT=~/projects/prefmem-libero        # ext4 副本, 含 upstream/LIBERO
export OPENPI_PY=$PREFMEM_ROOT/local_shims/serve_py_wrapper.sh   # SERVER_PY 替身（必须用 wrapper，裸 python 会跳过 config 注册）
cd "$PREFMEM_ROOT/scripts"
"$OPENPI_ROOT/.venv/bin/python" ../local_shims/run_local_a_arm.py <上面的相同参数>
```

  详见 `prefmem-libero/local_shims/README.md`；探活证据见 §11 B1b。

- 产物（脚本内固定）：`<out-dir>/cost.jsonl`、`<out-dir>/<task>/train.jsonl`、`<out-dir>/<task>/eval.jsonl`、`<out-dir>/<task>/{train,eval}_ep<N>.json`、`<out-dir>/lora/`（`save_pretrained`）。
- 历史 A 臂产物（**协议形状参考，非已训练基线**）：`prefmem-libero/results/qwen35_lora_rl8.json`（MD5 `171e01618fa5600f7ea57ae431b15078`）、`qwen35_lora_rl8_v2.json`（MD5 `59f0983ad08937ee44b94ebf333a26a9`）、`qwen35_lora_rl8_v3.json`（同上，与 v2 同字节）；三份均记录 8 任务、`n_train=12`、`n_eval=5`、seed 0。
  - ⚠ **训练有效性未确认**：三份文件的 `eval` 块均为 `updates=0`、`eligible=0`、`vlm_calls=0`、`desc_time_s=0`、`exec_time_s=0`、`update_time_s=0`，无法证明其来自已训练的 LoRA 策略。**不得直接当作"当前 LoRA FT"基线**；需 A 臂负责人确认来源后方可作对照锚点（FC13）。

### 5.2 邻接入口（执行器侧 LoRA，非 RL）

- `prefmem-libero/scripts/neutral_lora_batch.py`：`--specs <json> [--build] [--emit] [--install] [--train] --gpus 3,4,5 --only <ids>`
- 内部实际调用：`<openpi>/scripts/train.py <config_name> --exp-name ... --batch-size 1 --num-workers 0 --num-train-steps <steps, 默认 300> --save-interval 150 --checkpoint-base-dir /hhd2/.../checkpoints --assets-base-dir /hhd2/.../assets`（`neutral_lora_batch.py:95-119`），LoRA 变体固定为 `gemma_2b_lora` + `gemma_300m_lora`、`action_horizon=10`（同文件 `emit_config`）。
- 备选 RL 变体：`prefmem-libero/scripts/qwen_lora_rwr.py`（`--epochs 2 --lr 1e-4 --seed 0`，批式 RWR）。

### 5.3 A 臂不可移植点（已读，非推断）

| 位置 | 硬编码值 | 覆盖方式 |
|---|---|---|
| `qwen_lora_rl.py:30-32` | `SERVER_PY=/hhd2/yanghaobo/prefmem/openpi/.venv/bin/python`；`OPENPI_SRC`/`CLIENT_SRC` 同前缀 | **无** env 覆盖 |
| `qwen_lora_rl.py:107` | `--model /hhd3/yanghaobo/models/Qwen3.5-4B`（服务器原值） | CLI 可覆盖 → **本地已落盘**：ALWM `E:\Envs\JitRL_LoRA_Env\models\Qwen3.5-4B`、4090 `D:\Envs\JitRL_LoRA_Env\models\Qwen3.5-4B`（见 §11 B3a） |
| `neutral_lora_batch.py:19-21` | `ROOT`/`DATA_ROOT`/`ASSETS_ROOT` = `/hhd2/yanghaobo/prefmem/...` | **无** env 覆盖 |
| `neutral_lora_batch.py:96,99,104,109,110` | openpi 绝对路径 + `--checkpoint-base-dir /hhd2/...` | **无** |
| `controlled_loop_109.py:28-30` | `ROOT = os.environ.get("PREFMEM_ROOT", parents[1])`、`PY = os.environ.get("PREFMEM_PY", sys.executable)` | **有** env 覆盖（仅 ROOT/PY） |

---

## 6. B 臂（JitRL 测试期路径 · 等价替换 A 的 LoRA RL 槽位）

### 6.1 裁定（2026-11-01 两段，均 Asia/Shanghai）

- **07:44**：B 臂 = JitRL **测试期**路径（`z' = z + β·A`），作为 A 的 LoRA RL **训练槽位的等价替换**（不是错位路径）；须同 backbone（Qwen3.5-4B）、同 8 个 prefmem 任务、同执行器（299）。**07:36 的"必须权重训练"要求随之撤销**（该版要求已记为 superseded，见 §1）。
- **07:45 addendum**：B 臂必须点名「接地 LoRA 加载后的 **JitPi 偏好对齐**参考路径」。

### 6.2 参考路径定位结果（两仓分别盘点）

**（a）JitPi05 仓内：不存在偏好对齐 / LoRA 加载代码**（搜索证据，命令与命中数）

| 关键词 | `grep -rn -i <kw> src scripts docs plans reports` |
|---|---|
| `peft` / `LoraConfig` / `load_adapter` / `PeftModel` | **0 / 0 / 0 / 0** |
| `prefmem` / `preference_alignment` | **0 / 0** |
| `pairwise` | 18（全部出现在 `docs/issues/*` 的文字里，非代码路径） |

`docs/`(2 issues + 4 pdf)、`plans/`(1)、`reports/`(3) 中亦无该路径的设计或入口。

**（b）prefmem 仓内：存在完整的 JitRL 偏好选择路径**（这才是能同时满足 4B + 299 + prefmem 任务的实现）

| 角色 | 文件（ALWM 绝对路径） | 说明 |
|---|---|---|
| 控制器/循环 | `E:\Envs\JitRL_LoRA_Env\prefmem-libero\scripts\controlled_loop_109.py` | `--mission-dir/--mission-id` 驱动；`ROOT`/`PY` 可用 `PREFMEM_ROOT`/`PREFMEM_PY` 覆盖（`:28-30`） |
| 计划级 selector | `...\scripts\plan_jitrl_selector.py` | docstring 明示 `Q(s,plan)=mean G_pref`、`A=Q−V`、`π_β ∝ π_0(plan)·exp(β·A)`（测试期，无 optimizer） |
| 记忆版 selector | `...\scripts\plan_jitrl_memory_selector.py` | 同上 + memory |
| VLM 版 selector | `...\scripts\plan_jitrl_vlm_selector.py`、`...\plan_jitrl_vlm_text_selector.py` | 需要候选 JSON + 图像 + memory；文本键走 `vlm_text_key.describe()` |
| 候选生成 | `...\scripts\vlm_plan_proposer.py`、`vlm_online_candidates.py`、`vlm_candidate_generator.py` | 候选由 4B/VLM 产生（`candidates_*.json` 已有落盘件） |
| 执行器 | `...\scripts\serve_pi05.py` + `local_shims/serve_py_wrapper.sh` | 已实测 `--checkpoint 299` → `/healthz` 200（§11 B1b） |
| **现状审计** | `...\results\plan_level_jitrl_A_real.md` | 表格：`VLA executor = ⚠️ heuristic MotionGen, not π0.5`；`Next`：*replace heuristic executor with grounded LoRA π0.5 …* —— 该 "Next" 针对的是**旧 heuristic 执行路线**（`execute_selected_plan.py`）；新 hop `run_plan_with_vla.py` 已存在（B9a），环境半待 mission 包（B9b） |

### 6.3 注入矩阵（决定能否 READY）

| 要素 | 目标 | 现状 | 证据 |
|---|---|---|---|
| backbone 4B | 使用 `models\Qwen3.5-4B` | ✅ 数据就位；JitPi05 侧硬编码 2B 可**用 shim 覆盖**（patch `jitpi05.config.JITRL_QWEN_ID` 与 `jitpi05.jitrl.planner` 的已绑定名），prefmem 路径本身用 4B | §3/§11 B3a；`config.py:21`、`planner.py:61-63` |
| 执行器 299 | Jit 路径执行在 299 上 | ✅ prefmem 路径天然走 `serve_pi05 --checkpoint 299`；若走 JitPi05 `rlinf` 后端需 patch `_DEFAULT_{CHECKPOINT,SERVER_PYTHON,SERVER_SCRIPT}` + 注册 `local/prefmem_neutral` config | 实测：`pi05_libero` asset_id → **LOAD_FAIL**（缺 `assets/physical-intelligence/libero/norm_stats.json`）；`local/prefmem_neutral` → **LOAD_OK**(2) |
| 8 个任务 | 同 A 的 8 个 id | ❌ **不可达**：prefmem 任务 = `mission_info.yaml` 任务包（自定义彩色场景，如 `green_cream_cheese_1`），本地 `find . -name mission_info.yaml` = **0**；JitPi05 只能吃 stock libero suite（`jitrl.py:59-63` 的 `choices` 绑定 `JITRL_TASKS`，env 走 `LiberoEnvConfig(task=suite,task_ids=[…])`） | §6.4 / B9 |
| 偏好对齐入口 | 「接地 LoRA 后的 JitPi 偏好对齐」可运行 | ❌ 代码在 prefmem，但**输入缺失**且 VLM 描述器依赖外部 key（`vlm_text_key.load_key()` → `DEEPSEEK_API_KEY[_FILE]`） | B10 |

### 6.4 8 个任务 id（A 臂既有；B 必须同集）

```text
3-1_004_1_blue   3-1_006_1    3-2_032_7    3-2_032_8
3-2_097_8        3-2_097_10   3-9_118_1    3-10_117_1
```

这些是 prefmem 的 **mission id**（`controlled_loop_109.py:210-250` 以 `--mission-dir` 载入 `mission_info.yaml`），**不是** stock LIBERO 的 `suite + task_id`；本地 mission 包数量 = 0（B9）。

### 6.5 若 Lead 提供 mission 包与 VLM key，B 臂的**规定**入口形态（ALWM WSL，可复现模板）

```bash
export PREFMEM_ROOT=~/projects/prefmem-libero
export PREFMEM_PY=$HOME/projects/openpi/.venv/bin/python     # 或独立 runner venv
export OPENPI_ROOT=$HOME/projects/openpi
cd "$PREFMEM_ROOT/scripts"
# 1) 执行器（299）
OPENPI_ROOT=$OPENPI_ROOT PREFMEM_ROOT=$PREFMEM_ROOT \
  "$PREFMEM_ROOT/local_shims/serve_py_wrapper.sh" scripts/serve_pi05.py \
  --config pi05_libero_lora_neutral_v5init --checkpoint "$HOME/projects/ckpt/299" --port 8401 &
# 2) 候选（4B 产生；manifest 为既有 candidates JSON）
# 3) 偏好对齐选择（测试期 JitRL，z' = z + β·A）
"$PREFMEM_PY" plan_jitrl_vlm_text_selector.py \
  --mission-dir "$MISSION_DIR" --candidates "$CAND_JSON" \
  --memory "$MEM_JSON" --image "$IMG_PNG" --beta 1.0 --out "$OUT_JSON"
# 4) 闭环（no-memory vs memory 对照）
"$PREFMEM_PY" controlled_loop_109.py --mission-dir "$MISSION_DIR" --mission-id "$MISSION_ID" ...
```

> 该模板的**可行性已被局部证明**（299 服务 200；asset 解析 OK），但 **mission 包 + VLM key 缺失**，故不能声明 runnable（B9/B10）。

---

### 6.6 B selector v2：真实 Qwen3.5-4B logits（2026-11-01T10:3x，CPU，未触碰 A run）

- **问题**：v1 selector（`plan_jitrl_selector.py`）的 `pi0` 是**均匀先验**（只对 `β·adv` 做 softmax，不加载任何模型）→ 冷启动时 `Q≡0.5`、`A≡0`、永远选 index 0。
- **v2 实现**：新增 `prefmem-libero/local_shims/jit_selector_4b.py`
  - `pi0` 改为 **4B 首 token 分布**：prompt/取 token 逻辑**逐字复用** A 臂（`qwen_lora_rl.py:64-95` 的 `build_prompt / prepare_inputs / candidate_token_ids / policy_logits`，内联复制并注明出处，避免引入 peft）；
  - 更新式仍为 `z' = z + β·Â`（`Â` 来自 memory；冷启动为 0）；
  - 模型：`models/Qwen3.5-4B`，**4,539,265,536 参数**，`bfloat16`，**CPU**（`CUDA_VISIBLE_DEVICES=""`）。
- **运行时（隔离 venv，2026-11-01T10:49）**：`~/projects/jitrl_sel4b/.venv`（py3.11；`torch 2.15.0+cpu` / `torchvision 0.29.2+cpu` / `transformers 5.5.4`；`cuda=False`）；freeze **39 包 / sha256 `58d369f98e053b29fef7fcdf13209ea31563c10342e45bfd04f0d433a742ee46`**（清单 `artifacts/hf_pull_logs/sel_venv_freeze.txt`）。
- **8/8 mission 实测（隔离 venv 复跑）**（`artifacts/hf_pull_logs/sel8_newvenv.log`，逐 mission 决策 JSON `b_sel4b_newvenv_<mission>.json`）：

| mission | logit_std | pi0_std | pi0_entropy(nats) | Â var | selected |
|---|---:|---:|---:|---:|---:|
| libero_004_1_blue | 0.125 | 0.062 | 0.685 | 0.0 | 0 |
| libero_006_1 | 0.5625 | 0.255 | 0.557 | 0.0 | **1** |
| libero_032_7 | 0.5 | 0.231 | 0.582 | 0.0 | 0 |
| libero_032_8 | 1.9375 | 0.480 | 0.099 | 0.0 | **1** |
| libero_097_8 | 2.6875 | 0.495 | 0.029 | 0.0 | 0 |
| libero_097_10 | 0.125 | 0.062 | 0.685 | 0.0 | 0 |
| libero_118_1 | 3.0 | 0.498 | 0.017 | 0.0 | 0 |
| libero_117_1 | 1.4375 | 0.447 | 0.208 | 0.0 | 0 |

> 交叉核对：先在 `libero_rt311`（已冻结）得到一版（`sel4b_8.log`），隔离 venv 复跑结果除 `032_7`（1.625→0.5，torch 版本差异）外逐项一致；两版 logits 均非均匀。

- **判据**：4B 真实加载（路径 + 参数量 + shard MD5 见 §11 B3a）且 logits **非均匀**（std 0.125–3.0，entropy 0.017–0.685）→ 通过；**`Â` 方差 = 0 属合法冷启动**（README:88「每个 run 从独立空 memory 开始」），仅作 info。
- **运行时事件（补记）**：`libero_rt311` 在 10:01:12 装入 `accelerate==1.15.0`（dist-info mtime），供 4B `low_cpu_mem_usage` 加载路径使用；该 venv 现已被 Lead 冻结（§6.7），selector 的后续依赖只装入 `~/projects/jitrl_sel4b/.venv`。
- **已知缺陷（明示，不隐藏）**：
  1. 仓内**没有 Qwen 侧 LoRA 适配器**（`qwen_lora_rl.py` 只在训练结束时写 `runs/qwen35_lora_rl/lora`，A run 仍在进行）→ selector 用的是 **base 4B** 的 pi0；「grounded LoRA（299）」落地在**执行器**侧（`serve_pi05`）；若裁定要求 Qwen 也带 LoRA，需要 A run 产出适配器后再接。
  2. 冷启动 `Â≡0` ⇒ v2 当前选择 = `argmax pi0`；只有当 rollout 写入 memory 后 `Â` 才非零。
  3. 本 dry-run 为 **selector-only**（GPU 被 A 训练占满 15.5/16.4 GiB，不启动 299 执行以免争用）；执行 hop 已在 09:21 用 8/8 mission 独立验证过。

### 6.7 共享 venv 冻结声明（Lead 规则）

- **`~/projects/libero_rt311/.venv` 的角色 = 共享 A/B rollout client 运行时（LIBERO 执行侧）**：`robosuite==1.4.1` + `mujoco==2.3.7` + `numpy==1.26.4` + `torch 2.15.0+cpu` + `transformers 5.5.4` + `torchvision 0.29.2+cpu`（**`peft` 缺席**）。
- **状态：Lead 冻结，`no one may modify`**（omp 不得在其上 install/uninstall；只允许 `uv pip freeze` 导出）。
- **本轮开始前已存在的改动（记录，不回滚）**：`torch 2.15.0+cpu`（09:07 前后引入）、`transformers 5.5.4`（10:09）、`torchvision 0.29.2+cpu`（10:26）、`peft` 卸载（10:09，因与 tf5 冲突）。
- **freeze 指纹（规范方法，2026-11-01T11:0x 重算；venv 全程只读）**
  - **方法字符串（canonical，按 `ADDENDUM3_v2_lead_1101.md` §(i) 第 11 行）**：rt311 无 pip，故执行等价 = `uv pip freeze --python <venv>/bin/python` → `LC_ALL=C sort`（字典序）→ LF → UTF-8 → `sha256`。
  - **`libero_rt311`：`d9efcd2b5c63ead92e130c7d3ba62d74a21b729189b57f72e323928fd8c3a7fe`，95 行**（= Exp 实测值，逐字一致；状态 **待 RD 复核**，addendum3 明确"RD 与 Exp 相等前不作 reference、A 不得重启"）。
  - `jitrl_sel4b`：**`58d369f98e053b29fef7fcdf13209ea31563c10342e45bfd04f0d433a742ee46`，39 行**。
  - **交叉核对（`importlib.metadata`，同 95/39 个 dist）**：rt311 `45a0be688a6120ae23e9e2126594a60c527eb9016c5adb09a98a241bebd1ce8e`；sel4b `be2d37bc137d7cbf08f3300a8f4c4ac898bf8cd6acaa5ffa3edb9347a20a905b`。差异仅来自**命名规范化**（metadata 保留原大小写/连字符；uv/pip 归一化），包集合完全相同（95/39）。
  - `pip freeze --all --path <sp>`（临时 `uvx --from pip`，**未装入** venv）实测同 metadata 口径：rt311 `45a0be68…`、sel4b `be2d37bc…`。清单文件：`artifacts/hf_pull_logs/venv_freeze_after/libero_rt311.txt`、`artifacts/hf_pull_logs/sel_venv_freeze.txt`。
  - **旧值作废（void）**：
    - 本文件曾记 `…/228 包/ea309848…`（JitPi05 早期）与 `95 包/d9efcd2b…`（rt311 早期）：前者因后续装了 `msgpack` 等（见 §11 B1 注）变为 **230 包 / 88cb7c03…**；后者仍是当前值但因"未声明方法"而标记为**方法未记录**，以本节方法字符串为准。
    - **旧 `d9efcd2b…`（uv 格式）**：仍可复现，仅是**非 canonical 格式**，标记为"参考值"不再作为主值。
    - Exp ADDendum2 的 `c0a21a38…`：**当前 95 包集下不可复现（void）**。六组对照全部 ≠ 该值：
      pip 格式 sorted+LF `45a0be68…`、pip 未排序 `b6cfca0e…`、pip CRLF `3cfd39c3…`、pip+`pip==25.2` 行 `845969719…`；
      uv 格式 sorted+LF `d9efcd2b…`、uv 未排序 `132281c9…`、uv CRLF `64df3961…`、uv+`pip==…` `1318f248…`。
      ⇒ 差异来自**方法/时点**（Exp 环境含 pip 时 `pip freeze --all` 会多一行 pip、或排序/行尾不同、或计算早于 10:26 torchvision 安装），**不是包漂移**（10:48 与 10:57 两次冻结逐字节相同）。
  - **`accelerate` 事件**：`accelerate==1.15.0` 于 **10:01:12**（dist-info mtime）装入 `libero_rt311`；`jitrl_sel4b` 亦为 `accelerate==1.15.0`（10:49:05）。

```text
absl-py==2.5.1
accelerate==1.15.0
annotated-doc==0.0.5
anyio==4.15.1
attrs==26.1.0
bddl==3.6.0
certifi==2026.10.23
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
contourpy==1.3.3
cycler==0.12.1
defusedxml==0.7.1
easydict==1.13
evdev==2.0.0
fastjsonschema==2.22.2
filelock==4.2.4
fonttools==4.68.0
fsspec==2026.9.0
future==1.0.0
glfw==2.11.0
gym-notices==0.1.0
gym==0.26.2
h11==0.16.0
h5py==3.16.0
hf-xet==1.8.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.33.0
idna==3.20
imageio==2.38.1
iniconfig==2.3.1
jinja2==3.1.6
joblib==1.6.0
jsonschema-specifications==2025.9.1
jsonschema==4.26.0
jupyter-core==5.9.1
jupytext==1.19.6
kiwisolver==1.5.1
lazy-loader==0.6
llvmlite==0.50.0
markdown-it-py==4.2.0
markupsafe==3.0.3
matplotlib==3.11.2
mdit-py-plugins==0.6.1
mdurl==0.1.2
mpmath==1.3.0
msgpack==1.2.4
mujoco==2.3.7
nbformat==5.11.1
networkx==3.6.1
nltk==3.10.4
numba==0.68.1
numpy==1.26.4
opencv-python-headless==4.11.0.86
opencv-python==4.11.0.86
packaging==26.3
pillow==13.0.0
platformdirs==4.14.2
pluggy==1.7.0
psutil==7.2.2
pygments==2.21.0
pynput==1.8.2
pyopengl==3.1.10
pyparsing==3.3.3
pytest==9.2.0
python-dateutil==3.0.0
python-xlib==0.33
pyyaml==6.0.3
referencing==0.37.0
regex==2026.9.29
requests==2.34.2
rich==15.0.0
robosuite==1.4.1
rpds-py==2026.9.1
safetensors==0.8.0
scikit-image==0.26.0
scipy==1.17.1
setuptools==78.1.0
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
termcolor==3.3.0
tifffile==2026.3.3
tokenizers==0.22.2
torch==2.15.0+cpu
torchvision==0.29.2+cpu
tqdm==4.70.1
traitlets==5.16.1
transformers==5.5.4
typer==0.27.3
typing-extensions==4.16.0
tzdata==2026.6
urllib3==2.8.0
websockets==17.4
```

- 另两个共享 venv 本轮**未改动**（前后 freeze 一致）：`~/projects/openpi/.venv` `sha256=55b1afcb4b24fa017098a8d94480d0e84af1aa6d0837ccef6a6fb0dd8023a5b5`（242 包）；`~/projects/JitPi05-jitrl-vs-lora-acc/.venv` `sha256=88cb7c03704e2e36e192f2ba429dc2b0b443002326fdc405a8ae0faf5bd4b6ec`（230 包）。
- B 4B selector 专用 venv：`~/projects/jitrl_sel4b/.venv`，freeze `sha256=58d369f98e053b29fef7fcdf13209ea31563c10342e45bfd04f0d433a742ee46`（39 包）。


### 6.8 证据附录（Lead 11:19 规则：所有哈希现场重算）

- **sidecar**：`artifacts/xhost/EVIDENCE_20261101T1120.txt`（**31,777 B / MD5 `4e03a12cc53c804f1cf87c85a88948a8`**；现场重算——含每条哈希的**命令原文 + 原始输出**）。
- **本文件自身的哈希回执**：`artifacts/xhost/EVIDENCE_MANIFEST_HASH.txt`（对本 MANIFEST 的 `Get-FileHash -Algorithm MD5` + `(Get-Item).Length` 现场输出；本文件不自我引用该值）。
- 现场重算值（同一 sidecar）：
  - `rt311` freeze（`uv pip freeze` + `LC_ALL=C sort` + sha256）= `d9efcd2b…`，**95 行**；`jitrl_sel4b` = `58d369f9…`，**39 行**。
  - 交叉核对（`importlib.metadata`）：rt311 `45a0be68…`(95)，sel4b `be2d37bc…`(39) —— 差异仅命名规范化。
  - missions：**70 个 payload 文件**（排除 `mission_info.yaml.bak.*` 残留）md5-list = `98f372072f97e7003820bd6806aaa8744d61c286e868ece4916afb0d8fcc35fd`；含 2 个 `.bak` 残留时 **72 文件** = `aff1bf8f96fbaa3516a981707343278c0d9e62c77f3476fb29f90b28248cdbd8`（`./` 路径基准）。
  - assets：917 文件 / 721,608,920 B / md5-list（仓库根路径基准）= `05dada456f06a0f22df3bb2687e2fce8db5d4f328d5437cd37e853a6a08df4bf`。
  - ckpt299：28 文件 / 9,363,286,152 B / md5-list `977eb6e9…` / size-manifest `a9ee6806…`（`./` 基准）。
  - init-state（**旧 obs 口径，含图像 → 已被 §6.12 的 `libero-init-state/v1` canon 值取代，不可与 4090 对比**；`CUDA_VISIBLE_DEVICES=""`、`MUJOCO_GL=osmesa`、无 policy、seed 0）：`3-1_006_1` = `37908d70a3ee4ac7b94a1bbef309fe40d77e879ec22c9f7b60a944b2f1acf6f9`；`3-9_118_1` = `2068e0ff4cfd79080a7adcca5757e7e448eabb56f2bca04a47f3831131f65a2d`（GPU 前后 15,927 MiB 不变）。
  - ADDENDA：ADDENDUM3 `64c31a85…`；**ADDENDUM4 `0af879073f04e942da3d6f69ff631260`**（= Lead 期望 `0AF87907…`，`certutil -hashfile MD5` 现场计算；本 shell 的 `Get-FileHash` 不可用，方法已注明）。
- **UNVERIFIED（无法从当前真实文件复算者）**：`c0a21a38…`（Exp ADDENDUM2 值，六/八组方法变体均不匹配）、`ea309848…`（JitPi05 228 包旧基线）、`45a0be68…`/`be2d37bc…` 作为"主值"（现降为 metadata 交叉核对口径）。

### 6.9 ADDENDUM4（pending）与运行状态

- **ADDENDUM4_v2_lead_1115.md**：`0af879073f04e942da3d6f69ff631260`（现场计算），**状态 pending**；要点：**B 的记忆检索按 `task` 字段过滤，只用同任务条目**；记忆仍**跨任务累积**并每任务快照（与 ADDENDUM3 §(ii) 合并）。**未应用任何 Exp 补丁**。
- **action 输出一致性 = DEFERRED**（等 A run 结束；ALWM 现禁止任何 GPU 运行）。此前 11:11 的 fixed-input 探针输出 `actions_sha256=36cff5a97…`（shape [10,7]）**仅作偶发证据**：它实际连到 A 正在用的 8401 serve，按 Lead 裁决不作为跨机一致性结论。
- **B 结果在 action 一致性通过前 = provisional**；静态检查（freeze/资产/ckpt/init-state）已在 ALWM 通过 → 4090 静态复跑通过后可在 4090 以 **provisional** 运行 B。
- **ALWM 一致性运行状态 = PENDING_ALWM_GPU**（非失败）：11:11:16 的 xhost 运行 `EVAL_RC=1`，原因是 `--missions-root` 传成了 `series_3`（mission 目录实为 `perm_3_x/libero_*`）；已改用 `_resolve_dataset_dir` 规则映射，脚本留在 `artifacts/xhost/` 供 4090 复跑。
- **runner 已加固**：`run_xhost.sh` 改为专用端口 `8411` + `PID` 精确回收（`kill "$SERVE_PID"`），**已删除**任何 `pkill -f serve_pi05_local_config`。
  - **A 连续性（更正后的准确表述）**：A 旧 run（`0953`）已于 **11:03:10** 被 Exp 停止；当前 run 的 runner 44754 启动 **11:11:24**、serve 44902 启动 **11:11:37**，均**晚于**我们 11:11:16 的宽匹配 `pkill` → **没有杀到任何 A 进程**；危害仅为**潜在性**（若时序不同即命中）。加固已落地。


### 6.10 引用文件现场复算（Lead 11:27；MANIFEST 定版暂缓）

| 引用 | 现场值（命令原文+输出见证据 §10） | 期望 | 判定 |
|---|---|---|---|
| `ADDENDUM4_NOTE_v2_1119.md` | `30a45b5b92e1c046cc92cbf22f64f7ca` | `30A45B5B…` | **MATCH** |
| **B runner（gate v3，应用后）** | **无任何文件命中 `9f3ff54a…`**（全树 `run_local_b_arm*` 枚举：`a5738057…`/`51db6dab…`/`756b0568…`/`6f93e26e…`/`70a533a2…`/`71bff655…`；v3 patch 存在但未应用） | `9f3ff54a…` | **UNVERIFIED** |
| B runner（staged, gate v2）`patches/go_gate_stage_v2/…/run_local_b_arm.py` | `51db6dab94f7969118e61fed710cadd5` | `51db6dab…` | MATCH（**已被 gate v3 `9f3ff54a…` 取代**） |
| B runner（staged, gate v1）`patches/go_gate_stage/…/run_local_b_arm.py` | `a57380576ae058d816fd584caac11a67` | — | 记值 |
| B runner（live）`prefmem-libero/local_shims/run_local_b_arm.py` | `71bff655a824e199171dcf9717231af2` | — | 未打 gate 的 v1 |
| gate **v3** patch `run_local_b_arm_go_gate_failclosed_v3.patch` | `8de550dbc61f5cf2257336c7fb488e5e` | `8de550db…` | **MATCH** |
| gate v2 patch `run_local_b_arm_go_gate_failclosed_v2.patch` | `762fd07f3fbb598fe24f4d262f927588` | 前缀 `762fd07f` | **MATCH** |
| gate v1 patch `run_local_b_arm_go_gate_failclosed.patch` | `eeeabbcb71c0fd5e66d8587bfd09fdd4` | 前缀 `eeeabbcb` | **MATCH** |
| B patch（same-task filter）`run_local_b_arm_sametask_filter.patch` | `40d8d4b5887e4941bc9acdb70550bb34` | — | 记值 |
| B patch（symmetry t7）`run_local_b_arm_symmetry_t7.patch` | `eda28cdfc7a147b8572d9022200627b7` | — | 记值 |
| B patch（xtask logging）`run_local_b_arm_xtask_logging.patch` | `1f5d1b1d8a5aadbbc059789e5f4d9abb` | — | 记值 |

- **叠放顺序（v3 说明）**：`v1 eeeabbcb…` → `v2 762fd07f…` → `v3 8de550db…`，并叠加三个 B patch（`sametask_filter` / `symmetry_t7` / `xtask_logging`）。**v3 runner 的应用产物尚不存在** → 其哈希槽 UNVERIFIED（不猜测映射）。
- **GO_ARM_B 必填 5 字段**：`ADDENDUM4`、`MANIFEST`（本文件）、`NOTE`（`ADDENDUM4_NOTE_v2_1119.md`）、`PROVENANCE_0953`、`PROVENANCE_A3`；**PROVENANCE 仅按路径引用**（哈希记录在下游 `GO_ARM_B` / Lead）。
| 旧 run `artifacts/jitrl_vs_lora_acc/20261101T0953_seed0_5472f1f/PROVENANCE.md` | 仅按路径引用（**不在本文件记录其 MD5**） | — | 其哈希记录在下游（`GO_ARM_B` / Lead） |
| 新 run `artifacts/jitrl_vs_lora_acc/20261101T1111_seed0_a3/PROVENANCE.md` | 仅按路径引用（**不在本文件记录其 MD5**） | — | 其哈希记录在下游（`GO_ARM_B` / Lead）；§5.27 行含 `c415e1ab…` |
| B gate 相关（`GO_ARM_B` 等） | — | — | **PENDING**（Lead 未给终值） |

- **定版状态：FINAL（2026-11-01 12:4x；此后不再编辑）**。为避免循环哈希，**本文件不含任何 PROVENANCE 文件的 MD5**，仅按路径引用，其哈希记录在下游（`GO_ARM_B` / Lead）。早期 `ab9cf6a8…`（104,412 B）已 **SUPERSEDED**。
- **B（4090）= PROVISIONAL**；ALWM 一致性运行仍 **DEFERRED**。
- **ALWM 11:10 宽匹配 `pkill -f serve_pi05_local_config` 风险**：A runner 44754（lstart 11:11:24）/ serve 44902（11:11:37）均晚于该时刻 → **未命中任何 A 进程**；`run_xhost.sh` 已改专用端口 8411 + `kill "$SERVE_PID"`。
- **4090 交接**：`artifacts/xhost/HANDOFF_4090_stage2.md`（10,484 B / MD5 `3c7f956df855b0b5911495baa29d7ed7`，两侧一致）——**仅按路径引用**。
- 11:40 变更已并入本节：runner 基准改为 Gate v3（产物 UNVERIFIED）、v3 patch MATCH、v2/v1 与三 B patch 全部现场复算。


### 6.11 共同 base 299 与 A 臂 EVAL-ONLY 基线（Lead 11:55；MANIFEST 未定稿）

- **A 臂 = EVAL-ONLY**；**B 臂 = 299 + test-time JitRL**；两者**共用 base/executor 299**。
- **299（共享 base）**：`checkpoints/pi05_libero_lora_neutral_v5init/pi05_neutral_lora_v4_v5init/299` — **28 文件 / 9,363,286,152 B**；规范聚合（`cd 299 && find . -type f -exec md5sum {} + | LC_ALL=C sort -k2 | sha256sum`）= **`977eb6e9d52630877c938206ddf883f6b4e3aaa635ea7a7b286155355c219cc2`**，已在 E: 现场复现（证据 §16，无漂移；`a9ee6806…` 为 size-manifest，§4.2 表逐文件 28 行）。WSL `~/projects/ckpt/299` 逐文件 MD5 一致。
- **A 臂 eval ckpt：未选定**。历史候选人检索（E: 全量 / WSL `~` / .109 `~`+`/ssd*`+`/hdd*` / .110 `~`+`/hdd*`+`/nvme-ssd`）→ 7 个 `qwen35_lora_rl_*` adapter（84,972,248 B，base Qwen3.5-4B，r=16 α=32）：`8task`=`2f9ef649…`（8 任务，96 train 行=8×12）、`rwr`=`b7a57f55…`（0 train 行）、`smoke/smoke4/smoke5`=`e1ef4b71…`、`smoke2/smoke3`=`a57bb30b…`（单任务）、`qwen35_lora_rl_032_8` 无 `lora/`（未落盘）。**全部 executor 为 `/hhd2/...pi05_libero_lora_neutral_{006b/499,book/599,ketchup/599,milk/599}`，均非 299**；`results/qwen35_lora_rl8*.json` 缺失，且 4090 侧 `rl8` 汇总均 **updates=0**、本侧 `cost.jsonl` 为 0 B → **训练有效性未证**。故：唯一「有训练行」的 `8task` 也只能记为 **candidate（executor≠299，前提不符）**，**不由本文件选定**；A 臂 eval ckpt 留待 Lead 裁定（详见证据 §15–§17）。
- **范围声明（Lead 11:55 裁定，取代 11:54 版）**：**A 臂基线 = 已训练的历史 FC13 `qwen_lora_rl` 4B LoRA**；科学问 = **在 299 上的 test-time JitRL 能否取代该 LoRA**。**299 为两臂共享 base（MD5 见上）**。超出此范围的主张不在本文件效力内。
- **A 臂 eval ckpt = PENDING Lead**：候选 7 个 adapter（§15 全列），**无一在 executor 299 上训练** → 本文件**不选定**，由 Lead 裁定 FC13 归属/转换。
- **ADDENDUM5**：由 Exp 撰写；本文件**仅按路径引用**（不嵌哈希）。


### 6.12 4090 静态结果对齐 + D1（LIBERO 代码补丁）取证（Lead 12:0x；未定稿）

**两侧一致（MATCH）**
| 项 | 值 | 备注 |
|---|---|---|
| rt311 / sel4b freeze | `d9efcd2b…`（95 行） / `58d369f9…`（39 行） | 与 ALWM 逐字节一致（含尾 LF）；py 3.11.18 |
| 299 ckpt（规范方法） | **`977eb6e9d52630877c938206ddf883f6b4e3aaa635ea7a7b286155355c219cc2`** | 28 文件 / 9,363,286,152 B；E: 现场复现；4090 记法 `b2db1e74…` 为不同记法，语义一致（Lead 判定） |

**待对齐（PENDING，方法/根基准不同，暂不可比）**
| 项 | ALWM（规范方法） | 4090 报告 | 处置 |
|---|---|---|---|
| assets | `62230c0496f356189f06666fdd5ee0bb`（4090 口径：`relpath␠␠md5` 路径排序+尾 LF，MD5 of text；917 / 721,608,920 B） | `62230c04…` | **MATCH（双侧同法复算）**；旧 `996ff351…` 其他串行化口径作废 |
| series_3 | `7bf7d0b810434332f650aa547215a753`（75 / 547,306,200 B） | `a7231c4c354ec9ea3668976330a48b6f`（72 / 547,303,404 B） | **delta 完全对账**：+3 个 `mission_info.yaml.bak.*`（808+970+976=2,754 B）+ 同名文件差 42 B（见 D2）→ 2,796 B |
| 299 ckpt（4090 口径） | `b2db1e74a6ecf96d36843984cbf187ef`（28 / 9,363,286,152 B） | `b2db1e74…` | **MATCH**（与 ALWM 口径 `977eb6e9…` 同一清单两种串行化） |
| init_state（canon） | `BUNDLE_SHA256=245e6f4d19be6d61b7210ad149033126564eaf7227be513046aaffcb9e4e934c`（16/16） | 同值 | **MATCH** |

- **规范聚合方法（本文件唯一口径）**：`cd <root> && find . -type f -exec md5sum {} + | LC_ALL=C sort -k2 | sha256sum`。证据 §16 的 `7e17a9a0…` / `e3d946cb…` 为方法变体（绝对路径 / 仅 md5 列），**不作口径**。

**D1（4090 发现）— CLOSED**（补丁仅应用于 4090 clone，ALWM 侧代码未改；registry 50 → **991**；`object_keys_sha256=729fc8584310ca5d924d9bb875c57fc0156793ce8c55e10034b51647f596a73f`；两侧 init-state bundle 一致）
- 仓库：WSL `~/projects/prefmem-libero/upstream/LIBERO`，`HEAD = 8f1084e3132a39270c3a13ebe37270a43ece2a01`（与上游 pin 一致）。
- 工作树并非上游原状：`tracked_changed = 1083`、`untracked_total = 807`（其中 **135 个未跟踪 `.py`**）。
- 包：`artifacts/xhost/libero_patch/`（141 文件 / 12 MB），含 `HEAD.txt`、`status_porcelain.txt`、`libero_diff_tracked.patch`（10,264,788 B）、`untracked_py/…`（135 个）、`MD5SUMS.txt`、`summary.txt`。
- **D1 注册类确实由补丁引入**：`green_butter` / `green_cream_cheese` / `blue_ketchup` / `red_yellow_book` 在 tracked diff 与未跟踪 `.py` 中均出现（`libero/libero/envs/objects/__init__.py` 等）；即 4090 直连上游会缺类，ALWM 侧靠该补丁补齐。
- 包锚点 MD5（142 文件）：`MD5SUMS.txt`=`860075da254215f56139850321d8a79d`（自列行已剔除，重算安全）、`d1_registration_grep.txt`=`b5bc0739b6249714da483e495e6e6657`、`libero_diff_tracked.patch`=`c46163be0705112b4d9d2cf1aa6645c4`、`HEAD.txt`=`a0663d4e028c78cad8f61750a2d792eb`、`status_porcelain.txt`=`8a4acfc8cc13edf236150f03eb73b70e`、`summary.txt`=`6166117b5c127dd7c93eb336db3d0227`。
- **D2（新，阻塞跨臂可比性）— 任务 `3-2_032_7` 指令文本两侧不同**：ALWM `mission_info.yaml`（850 B / `a443eadb8ed8fbe7fe06d9e02df09a49`）= `"Put the tomato sauce in the basket."`；4090（808 B / `f2ef2f7ee97e145b88074122c205d5ea`）= `"Put a sauce in basket"`；`language_instructions` / `optimized_instruction` / `libero_instruction` 三字段均不同，`bddl_file` 相同（init-state 未受影响，16/16 仍一致）。**A 在 ALWM、B 在 4090 → 同一任务不同 prompt，D2 解决前 A/B 数值不可比**；统一口径由 Lead 裁定。
- **注册类落点**（`@register_object`）：`hope_objects.py` 的 `BlueKetchup` / `GreenButter` / `GreenCreamCheese`，`turbosquid_objects.py` 的 `RedYellowBook` / `RedYellowBookSide`；运行期由 `3-1_004_1_blue` 的 `declared_object_types` 含 `green_cream_cheese` 且 `status=OK/deterministic=true` 证实。
- **v1 脚本缺陷（已上报，未改脚本）**：`--layout naive|both` 在 `init_state_canon_v1.py` 第 309–310 行把目录名 `no_pref_layout` 直接当 layout 传入，而 `LAYOUT_DIR`（第 110 行）无此键 → 第 210 行 `KeyError`；工作方式为显式 `--mission TID:REL:naive`（`normalize_layout` 正确映射）。
- **layout 语义**：`pref` 与 `naive` 的 bddl `:init` 段**逐字节相同**（抽查 `perm_3_9/libero_118_1` 与 `perm_3_1/libero_006_1`，init 段 diff=0；整文件差 12 / 7 行 = 仅 goal/偏好段）。故 6 个任务两 layout 的 `init_state_sha256` 相同应读作 **byte-identical serialized state — mechanism unexplained**（`:init` 段相同不蕴含状态必然相等：`3-9_118_1` 的 `:init` 亦相同却哈希不同，差异来自 `:objects`/goal 段序导致的 qpos 布局位移）；其中 `3-2_097_8`、`3-2_097_10` 的 model XML 本身不同（47,403 vs 47,405 B；50,422 vs 50,421 B），已如实标注，非解析错位。


**规范 init-state（libero-init-state/v1）— ALWM 现场结果**

- 脚本校验：`artifacts/xhost/init_state_canon_v1.py` = `d64844565c283781ebdca79553ab2ab8`（= Lead 提供 `D64844565C283781EBDCA79553AB2AB8` **MATCH**）。
- 运行：`CUDA_VISIBLE_DEVICES="" MUJOCO_GL=osmesa`，rt311 `python 3.11.18` / np 1.26.4 / robosuite 1.4.1 / mujoco 2.3.7，seed 0，`repeat=2`（两次构造一致），**不加载 policy**；16 targets = 8 mission × {pref, naive}。
- 结果：**16/16 OK**；`object_types=991`；`object_keys_sha256=729fc8584310ca5d924d9bb875c57fc0156793ce8c55e10034b51647f596a73f`；**`BUNDLE_SHA256=245e6f4d19be6d61b7210ad149033126564eaf7227be513046aaffcb9e4e934c`**。
- 侧车：`artifacts/xhost/init_state_canon_ALWM.json`（613,771 B / MD5 `49ff4ec855a405c0f19a892d7703ad27`）；`artifacts/xhost/init_state_model_xml_ALWM.json`（3,968 B / MD5 `60e2c632b3e7fb1f52b78cf9c0fa59b3`；**非 v1 规范**，为 `sha256(env.sim.model.get_xml())` 派生，含 meta 块）。

| task | pref init_state_sha256 | naive |
|---|---|---|
| 3-1_004_1_blue | `bac173d5c8384225baa2925d08b49f6cba286e5cf11dec5af01cadc91f6544b2` | 同 pref |
| 3-1_006_1 | `8e5cad4212395e07ea1155fe3cc551ac521db629132a27c55aa66b62380abaca` | 同 pref |
| 3-2_032_7 | `2fa239ac63d2a527341fba5c9713f8a4733b6955423028e942bf772ead05255f` | 同 pref |
| 3-2_032_8 | `83e810923b735ae21b13b353cac88581e9bcdd7c43e7c79fc28357b67ba65659` | 同 pref |
| 3-2_097_8 | `5722312b456f42150898c4f2a6acf3eb19941af49cb7e095acfd4b93019cfdf3` | 同 pref |
| 3-2_097_10 | `3e6b50c93a9b8e740fab192e50e0b0c138124d51aab922b5b04aa18107ba21b4` | 同 pref |
| 3-9_118_1 | `b4bf831331f66d7a2c56f287becd5f7942e002822f487b48d81e09f5424aec3a` | `2da528837420cc5d866a16f3fd51c787efb12afc5a77f6127564edec5c941f95` |
| 3-10_117_1 | `642857b4ebf87b48cb32167ae356d92cd17a1613e69adea86e3dd789e21a9d74` | `04ddbfb67516a5824832691f818920698d1bea5e2b74058ed49eee2ec820c7fb` |


## §17 FINAL — 取代 §6.12 的 PENDING 行与 11:54 版范围句（Lead 11:55 裁定）

> 本块为最终口径；凡本文档更早处与之冲突（含 §6.12 表格中标注 PENDING/待比对的行、以及已删除的 11:54「不是取代 RL」句）一律以本块为准。

### 17.1 范围（取代 11:54）
- **A 臂基线 = 已训练的历史 FC13 `qwen_lora_rl` 4B LoRA**；**B 臂 = 299 + test-time JitRL**。
- **科学问**：在 **299** 上的 test-time JitRL **能否取代该 LoRA**。
- **299 为两臂共享 base/executor**：规范聚合（`cd 299 && find . -type f -exec md5sum {} + | LC_ALL=C sort -k2 | sha256sum`）= **`977eb6e9d52630877c938206ddf883f6b4e3aaa635ea7a7b286155355c219cc2`**（28 文件 / 9,363,286,152 B）；4090 记法（`relpath␠␠md5` 路径排序+尾 LF，MD5 of text）= **`b2db1e74a6ecf96d36843984cbf187ef`** —— **同一清单两种串行化，两侧 MATCH**。

### 17.2 A 臂 eval ckpt = PENDING Lead（7 个 adapter 全列，无一 post-RL on 299）
- 证据文件：`artifacts/xhost/EVIDENCE_LORA_CANDIDATES_ALWM.txt`（**4,561 B / MD5 `28a07a1b3ee7839e5a085f48f942057e`**，含每个 run 的 `size/mtime/md5sum` 原始输出、`adapter_config.base`、`r/alpha`、train 行数、`cost.jsonl` 字节、executor 还原行）。
- 全部位于 `.109`(=baode)`/ssd0/yanghaobo/prefmem/from110/prefmem-libero/runs/`；base 均 `Qwen3.5-4B`，`r=16 α=32`，adapter 大小均 **84,972,248 B**。
- **排除**：今日 09:07 起作废 run 与 run a3 产物（该 root 下无 a3）。

| run | adapter md5 | mtime | train | executor | 分类 |
|---|---|---|---|---|---|
| `qwen35_lora_rl_8task` | `2f9ef649adc5678fcf0ab68b869d375d` | 2026-10-29 18:12:23 | 8 目录 / **96 行** | 006b/499、book/599、ketchup/599、milk/599 | RL 有训练行，**executor ≠ 299** |
| `qwen35_lora_rwr` | `b7a57f55acef1abbf5fd07e59b43e72b` | 2026-10-30 04:12:44 | 0 / 0 | 同上四者 | RWR，无 train 行 |
| `qwen35_lora_rl_smoke` | `e1ef4b71064a8f6a10b01cadf0b64dee` | 2026-10-29 16:06:40 | 1 / 3 | ketchup/599 | 冒烟 |
| `qwen35_lora_rl_smoke2` | `a57bb30b6b40d718867a603d1f8d6f6f` | 16:07:31 | 1 / 2 | ketchup/599 | 冒烟 |
| `qwen35_lora_rl_smoke3` | `a57bb30b6b40d718867a603d1f8d6f6f` | 16:10:38 | 1 / 2 | ketchup/599 | 冒烟（与 smoke2 同字节） |
| `qwen35_lora_rl_smoke4` | `e1ef4b71064a8f6a10b01cadf0b64dee` | 16:14:34 | 1 / 3 | ketchup/599 | 冒烟（与 smoke 同字节） |
| `qwen35_lora_rl_smoke5` | `e1ef4b71064a8f6a10b01cadf0b64dee` | 16:17:26 | 1 / 3 | ketchup/599 | 冒烟（与 smoke 同字节） |
| `qwen35_lora_rl_032_8` | **无 adapter**（无 `lora/`；train 11 行） | — | 1 / 11 | ketchup/599 | 未落盘 |

- **`cost.jsonl` 全为 0 B**，4090 侧 `rl8` 汇总 **updates=0** → 训练有效性未证；**唯一有 8×12 train 行的 `8task` 也不在 299 上训练**。
- **≥2 个候选 / 无一满足 post-RL on 299 → 不选**：A 臂 eval ckpt = **PENDING Lead**。

### 17.3 Gate v3 叠层与 GO_ARM_B
- runner **gate v3** 目标值 `9f3ff54a54c31631d38d8db86b65bcc6`；v3 patch `run_local_b_arm_go_gate_failclosed_v3.patch` = **`8de550dbc61f5cf2257336c7fb488e5e`**，叠于 **v2 `762fd07f3fbb598fe24f4d262f927588`** + **v1 `eeeabbcb71c0fd5e66d8587bfd09fdd4`** + 三个 B patch（`sametask_filter 40d8d4b5…` / `symmetry_t7 eda28cdf…` / `xtask_logging 1f5d1b1d…`）；**`51db6dab…` 已被 v3 取代**。v3 **应用产物在树中不存在** → 该 runner 槽 UNVERIFIED。
- **GO_ARM_B 必填 5 字段**：`ADDENDUM4`（`0af879073f04e942da3d6f69ff631260`）、`MANIFEST`（本文件）、`NOTE`（`ADDENDUM4_NOTE_v2_1119.md` = `30a45b5b92e1c046cc92cbf22f64f7ca`）、`PROVENANCE_0953`、`PROVENANCE_A3`。
- **PROVENANCE 与 ADDENDUM5 仅按路径引用**，不在此记 MD5（哈希记录在下游 `GO_ARM_B` / Lead）。

### 17.4 跨机静态对齐（取代 §6.12 的 PENDING 行）
| 项 | ALWM | 4090 | 结论 |
|---|---|---|---|
| freeze | `d9efcd2b…`(95) / `58d369f9…`(39) | 同值 | **MATCH**（含尾 LF，py 3.11.18） |
| assets | `62230c0496f356189f06666fdd5ee0bb`（917 / 721,608,920 B） | 同值（同法） | **MATCH** |
| 299 | `977eb6e9…`（ALWM 记法）/ `b2db1e74…`（4090 记法） | 同 | **MATCH** |
| init-state canon | `bundle_sha256=245e6f4d…4e934c`，16/16 OK | 同值 16/16 | **MATCH** |
| D1（LIBERO 注册类） | 补丁存在（`libero_patch/` 142 文件） | 已应用补丁 | **CLOSED**（registry 50→991；`object_keys_sha256=729fc858…`） |

**series_3 delta = 精确对账，无残余**
- 仅 ALWM 有的 3 个文件（原子写备份）：`perm_3_2/libero_032_7/mission_info.yaml.bak.57459…`(808) + `perm_3_2/libero_097_8/mission_info.yaml.bak.42504…`(970) + `…bak.42683…`(976) = **2,754 B**。
- **同名但内容不同**：`perm_3_2/libero_032_7/mission_info.yaml` —— ALWM **850 B**（`a443eadb8ed8fbe7fe06d9e02df09a49`，`"Put the tomato sauce in the basket."`）vs 4090 **808 B**（`f2ef2f7ee97e145b88074122c205d5ea`，`"Put a sauce in basket"`）；**4090 版本 = ALWM 的 `.bak.57459`**。
- `2,754 + 42 = 2,796 B` = 观测差（75/547,306,200 vs 72/547,303,404）→ **完全收敛**。
- **任务内容未变**：`pick_up_the_blue_ketchup_and_place_it_in_the_basket.bddl` 两侧 sha256 **`aaf174b222fe5be4c3d4d9db783e0c6c3f010e890cd777f90a2bed3f786522cf`**（1,435 B）**完全一致**；init-state `2fa239ac…` 亦一致。
- **D2（YAML `blockers: [D2]`）**：仅 **instruction 文本**在两侧不同（`language_instructions` / `optimized_instruction` / `libero_instruction` 三字段）→ **A(ALWM) 与 B(4090) 在同一任务使用不同 prompt，D2 解决前 A/B 数值不可比**。

### 17.5 状态与保留项
- **B（4090）= PROVISIONAL**；**ALWM 一致性 = DEFERRED**；2×5 判分 = DEFERRED。
- **ALWM 11:10 宽匹配 `pkill -f serve_pi05_local_config` 风险**：A runner 44754（lstart 11:11:24）/ serve 44902（11:11:37）均晚于该时刻 → **未命中任何 A 进程**；`run_xhost.sh` 已改专用端口 8411 + `kill "$SERVE_PID"`。
- 保留：§6.6 的 accelerate 1.15（10:01）；`d9efcd2b` vs `c0a21a38` 对账与方法串（`uv pip freeze`，待 Lead 终值）；ADDENDUM3 取代 B10。
- 证据 sidecar：`artifacts/xhost/EVIDENCE_20261101T1120.txt`（**31,777 B / MD5 `4e03a12cc53c804f1cf87c85a88948a8`**，§1–§21）。
- **4290 交接**：`artifacts/xhost/HANDOFF_4090_stage2.md`（10,484 B / `3c7f956df855b0b5911495baa29d7ed7`，两侧一致，按路径引用）。
- 早期 `ab9cf6a8…`（104,412 B）与同轮 `f08d76d2…`（115,810 B）均 **SUPERSEDED**。
- **本文件 FINAL；此后不再编辑。**

## 7. 协议对齐（按裁定强制固定；B6/B7 由此 RESOLVED）

裁定把"同任务集 + 同 backbone"从"待协商"变成"必须满足的固定项"，因此下表不再是"差距清单"，而是**两臂共享的固定合同**：

| 维度 | 固定值（两臂相同） | 依据 |
|---|---|---|
| 任务集 | 8 个 id：`3-1_004_1_blue`、`3-1_006_1`、`3-2_032_7`、`3-2_032_8`、`3-2_097_8`、`3-2_097_10`、`3-9_118_1`、`3-10_117_1` | Haobo 裁定（同任务集）；清单来源见 §6.4 |
| backbone（高层） | `Qwen3.5-4B`（`E:\Envs\JitRL_LoRA_Env\models\Qwen3.5-4B`，43 文件 / 9,342,908,952 B） | 裁定（同 backbone）＋ B3a 落盘证据 |
| 执行器（低层） | `pi05_neutral_lora_v4_v5init/299` 线（28 文件 / 9,363,286,152 B / md5-list `977eb6e9…`），`serve_pi05` 探活 200 | 裁定（基线=grounded LoRA 299 线）＋ B1b 探活 |
| 种子 | `0` | 裁定（不换种子） |
| 训练预算 | 每 task `12` 训练 episode（A 侧既有） | 裁定（同步数）→ B 侧须同名语义，见 B8 |
| 评测预算 | 每 task `5` eval episode | 裁定（同评测） |
| 主指标 | 任务执行准确率（A 侧字段 `sr` / `strict` / `compliance`） | 裁定（对照任务执行准确率） |
| 更新规则 | **A：LoRA+REINFORCE 权重更新；B：JitRL 测试期 `z'=z+β·A`（无梯度）** | 裁定 07:44（等价替换槽位） |

**唯一剩余差异 = 更新规则本身**（这正是被检验的自变量）。除此之外任何差异（任务、backbone、执行器、种子、预算、指标）都视为违约。

### 7.1 对照叙事（钉死，CN + EN）

- **CN**：论文主张 JitRL 可在**不更新权重**的前提下替换 RL。本实验的对照**不是换任务套件**，而是在**同一 grounded LoRA 训练设定**内替换更新规则：
  - **A = 接地 LoRA 的 RL**：`Qwen3.5-4B` 上 `qwen_lora_rl` 做权重更新 + 冻结执行器 `299`；
  - **B = 接地 LoRA 之后的 JitPi 偏好对齐**：测试期 `z' = z + β·A`（`π_β ∝ π_0·exp(β·A)`），**无新增权重训练入口**。
  - **评测口径**：同 8 个 prefmem 任务 / 同 backbone / 同 `299` 上的**任务执行准确率** A vs B —— 以此对比证明 JitRL 在 VLA 上是否有效。
- **EN**：The JitRL paper claims it can replace RL **without weight updates**. This line's contrast is **not a different task suite**; it swaps the update rule inside the **same grounded-LoRA setup**: A = grounded LoRA RL (weight updates on `Qwen3.5-4B` + frozen executor `299`); B = post-grounded-LoRA JitPi preference alignment (test-time `z'=z+β·A`, no new weight-train entry). Metric: task accuracy A vs B on the **same 8 prefmem tasks / same backbone / same 299**.

> 与 §1.0 动机一致：本对比要回答的是**可替换性**（能否整体替换 VLA RL 的更新规则），而非"偏好训练是否有效"。

---

## 8. GPU 占位（实测）

| 主机 | GPU | 总显存 | 已用（基线） | 备注 |
|---|---|---:|---:|---|
| ALWM18R2（本机） | 0 · `NVIDIA GeForce RTX 4090 Laptop GPU` | 16,376 MiB | 1,195–1,318 MiB | 桌面占用 ~1.2 GB |
| 2026-552A-4090 | 0 · `NVIDIA GeForce RTX 4090` | 24,564 MiB | 12,588 MiB | 桌面（dwm/qq/chatgpt）占用 ~12.6 GB；本机不参与执行（B5） |

- **A 臂 299 服务实测峰值**：**13,721 / 16,376 MiB**（2026-11-01T03:20 探活；停服后回落 1,195 MiB）→ 余量 ≈ **2.6 GB**。
- 结论：A 臂（299 服务 + LoRA runner）**无法与另一个 4B 级策略同驻**本机 16 GB GPU；B 臂若需同时驻留 backbone + 执行器，必须先做显存预算方案（属 B8 落地的一部分）。

---

## 9. 输出路径（规定；**本次未创建任何目录**）

统一约定（两机同构，`<root>` 分别为 `E:\Envs\JitRL_LoRA_Env` / `D:\Envs\JitRL_LoRA_Env`）：

```text
<root>\artifacts\jitrl_vs_lora_acc\<run_id>\
  arm_a\                       # LoRA RL（A）
    cost.jsonl
    lora\                      # save_pretrained
    <task_id>\train.jsonl
    <task_id>\eval.jsonl
    <task_id>\{train,eval}_ep<N>.json
  arm_b\                       # JitRL 测试期（B）——入口待 B9/B10，下列为**规定**形态
    <mission_id>\decision_ep<N>.json   # selector 决策（Q/V/A、β、被选 plan；字段见 plan_jitrl_*）
    <mission_id>\execute_ep<N>.log     # 299 执行日志（hop 落地后）
    <mission_id>\eval_ep<N>.json       # 评测记录（sr/strict/compliance，字段与 A 对齐）
    memory.json                        # 跨 episode 记忆（prefmem TextMemory 格式）
    metrics.json
    run_summary.json
    progress.json
    # 注意：B 是测试期方法，**不产出权重**（无 ckpt/、无 optimizer state）
  PROVENANCE.md                # 必须包含：两臂 commit/seed/命令行全文/ckpt 299 复算指纹/GPU 快照
```

- `run_id` 规范：`20261101T<HHMM>_seed0_5472f1f`（本次刻意未占用任何 run_id）。
- 禁止：写入任何 `full74*`、`.109 full74v3`、或其他既有 `runs/` 历史目录（FC9）。
- 两机产物**不得**互相覆盖；镜像/汇总由 Lead 指定单写者。

---

## 10. fail_closed_assertions（fail-closed 断言，15 行 / FC1–FC15，开训前必须逐条实测通过）

| ID | 断言 | 检查方式 | 现状 |
|---|---|---|---|
| FC1 | `STATUS` 与 §11 一致：§11 的 **B-list blocker** 任一未标 `RESOLVED` → 必须为 `BLOCKED` | 人工+reviewer | ✅ §11 B-list 全数 RESOLVED（B9 已解），故 `status: READY` 成立；YAML `blockers: [D2]` 属**跨臂可比性 gate**（D2 只封锁 A/B 数值对比，不属 §11 B-list），已在 §17.4 显式声明 |
| FC2 | 两机 worktree `HEAD == 5472f1f43e84750cb2b0760b17b5853cd788eddb`，且 `git status --porcelain` **仅含**：本 MANIFEST + 既有 `M docs/papers/CycleVLA.pdf`（MD5 `96dc4ba22b674a74d12b2a3fa0672a15`，两机同字节，须由 Lead 认定是否接受为基线）；`.venv/` 已 gitignore；`third_party/{openpi,libero-pro}` 必须停留在 gitlink 指定 commit（见 B4），否则视为污染 | `git -C <wt> rev-parse HEAD`；`git status --porcelain`；`git -C third_party/openpi rev-parse HEAD` | ⚠ 已 pin，待认定 |
| FC3a | 两机 299 目录 28 文件 MD5 全等于 §4.2 | `find . -type f -exec md5sum {} + \| sort -k2`；对照 §4.2 | ✅ 实测通过 |
| FC3b | `openpi_assets` 副本与 299 内 `norm_stats.json` 同 MD5；否则记录差异 | `md5sum` 两处 | ✅ 实测两处同为 `03d102a7…`，双机一致 |
| FC4 | ALWM WSL 内 `.venv` 存在，`uv run jitpi05-doctor` 退出码 0（ALWM-only 范围） | `wsl -d Ubuntu-24.04 -e bash -lc 'cd ~/projects/JitPi05-jitrl-vs-lora-acc && uv run jitpi05-doctor'` | ✅ 实测退出码 0（6.9 GB venv / **230 包**；10:48 基线 `88cb7c03…`） |
| FC5 | `uv sync --frozen` 后 `git diff --quiet -- uv.lock`（依赖无漂移） | `git diff --exit-code uv.lock` | ✅ 实测无漂移（WSL checkout 干净） |
| FC6 | 两臂 `--seed` 均为 `0`（裁定"不换种子"） | 两侧命令行/trace | ⚠ 契约已定，B 侧待跑 |
| FC7 | 两臂评测预算均为每 task `5` episode | 命令行+产物 `n_eval`/`n` | ⚠ 契约已定，B 侧待跑 |
| FC8 | ALWM 目标 GPU 余量满足预算：A 臂 299 服务实测峰值 **13,721/16,376 MiB → 余量 ≈2.6 GB**；B 臂若与 A 同机顺序执行，须给出"先停 A 服务再跑 B"的调度或显存方案 | `nvidia-smi` 快照写进 `PROVENANCE.md` | ❌ 未解（无调度/显存方案） |
| FC9 | 输出目录不存在（本次新建），且路径不含 `full74`/`full74v3` | `test -e`；字符串断言 | ⚠ 待定 |
| FC10 | 两臂 task 集合逐 id 相等：裁定固定为同 8 个 mission id（§6.4），B 侧须逐 id 断言 | id 集合 diff | ❌ **B9**（8 个 mission 包本地为 0，B 无法载入） |
| FC11 | 两臂高层 backbone 同一：`Qwen3.5-4B`（43 文件 / 9,342,908,952 B） | 读 CLI + 权重 MD5 | ⚠ 契约已定；B 侧可传（prefmem 路径本身用 4B），未执行 |
| FC12 | 两臂低层执行器同一：`299` 线（md5-list `977eb6e9…`，`serve_pi05` 探活 200） | `serve_pi05 --checkpoint` + 校验和 | ⚠ A 侧已实测；**B 侧服务器半已验**（healthz 200 + `actions[10,7]`），**env-half 未通过**（B9b-5） |
| FC13 | 若引用 `results/qwen35_lora_rl8*.json` 作 A 臂基线，须由 A 臂负责人确认来自**已训练** LoRA（该文件 `updates=0/eligible=0/vlm_calls=0`） | 负责人书面确认 + trace 交叉核对 | ⚠ 未确认（仅作任务清单/形状参考） |
| FC14 | B 臂**测试期**入口可运行：`plan_jitrl_*` + 299 服务能对同一 mission 产出决策 JSON 与执行日志（无梯度） | 一次 dry-run（非训练）：决策 JSON + `/healthz` 200 + 执行日志 | ❌ **B9b-5**（env-half：`green_butter` 未注册 / 环境无法 reset） |
| FC15 | B 臂口径可命名且可复算：Q/V/A=Q−V、`π_β ∝ π_0·exp(β·A)`（或声明改用主结果的 argmax-Q 口径）、β 取值、决策 JSON 字段 | 源码 + 决策 JSON 抽样 | ❌ **B9/B10**（入口输入缺失，无法产出可复算样本） |

---

## 11. BLOCKERS（**全部 RESOLVED**：B1–B10；`STATUS=READY` 的前置证据见 B9 与 §12）

### B1 · 依赖安装｜**RESOLVED（2026-11-01T02:03:09+08:00 venv 落定；doctor 随后 EXIT=0）**
- 证据（WSL `Ubuntu-24.04`，ext4 checkout `~/projects/JitPi05-jitrl-vs-lora-acc`）：
  - `HEAD = 5472f1f43e84750cb2b0760b17b5853cd788eddb`（与 worktree 同 commit）。
  - `uv sync --frozen` **退出码 0**（日志 `/tmp/sync.log`；无 `error`/`failed` 行；`uv` 版本 0.13.6，CPython 3.12.3）。
  - `.venv` = **6.9 GB**，`uv pip list --format freeze` 共 **230 包**，清单 SHA-256 `88cb7c03704e2e36e192f2ba429dc2b0b443002326fdc405a8ae0faf5bd4b6ec`（02:03 基线曾为 228 包 / `ea309848…`；此后为 LIBERO client 装入 `msgpack` 等 → 230）；关键版本：`torch==2.11.0+cu130`、`triton==3.6.0`、`flash-linear-attention==0.5.2`、`lerobot==0.6.0`、`transformers==5.5.4`。
  - `uv run jitpi05-doctor` **退出码 0**，输出：`platform=Linux-6.18.54.1-microsoft-standard-WSL2-x86_64`、`pytorch=2.11.0+cu130`、`cuda_runtime=13.0`、`device=NVIDIA GeForce RTX 4090 Laptop GPU`、`compute_capability=8.9`、`bfloat16=true`、`fla_backend=cuda`。
  - **依赖无漂移（FC5 通过）**：`git status --porcelain` 空、`git diff --stat -- uv.lock` 空。
- 历史失败证据（保留，说明为何必须 WSL）：Windows 原生 `uv sync --frozen` 在 ALWM 与 4090 均退出码 2 —— `triton==3.6.0` 无 `win_amd64` wheel（`pyproject.toml:14` 的 `flash-linear-attention[cuda]` 依赖链），修复需改 `pyproject.toml`（禁止）。
- 适用范围：**仅 ALWM WSL**（B5 ALWM-only 裁定）；4090 Windows 侧不建 env。

#### B1b · A 臂（prefmem）运行时环境｜**RESOLVED（2026-11-01T03:21:18+08:00，ALWM WSL2）**
- **上游代码（条件 1）**：`~/projects/prefmem-libero/upstream/LIBERO`（ext4），`git rev-parse HEAD` = `8f1084e3132a39270c3a13ebe37270a43ece2a01`。
  ⚠ **仓内无 pin**（prefmem-libero 无 `.gitmodules`，`prefmem-libero/upstream` 在服务器上亦非子模块）→ 该 SHA 是"本次 clone 的上游 HEAD"，属 **provenance UNKNOWN，需 Lead 认定**；重跑时须固定该 SHA。
- **openpi venv（条件 2）**：`~/projects/openpi` @ `15a9616a00943ada6c20a0f158e3adb39df2ccac`（pinned gitlink），`uv venv --python 3.11` + `uv sync --frozen` → **`OPENPI_SYNC_EXIT=0`**，`.venv` **7.7 GB / 242 包**，freeze SHA-256 `55b1afcb4b24fa017098a8d94480d0e84af1aa6d0837ccef6a6fb0dd8023a5b5`；关键版本 `torch==2.7.1`、`transformers==4.53.2`、`jax==0.5.3`、`flax==0.10.2`、`orbax-checkpoint==0.11.13`、`peft==0.21.2`、`accelerate==1.15.0`。
- **执行器探活（条件 3）**：ckpt 复制到 ext4 `~/projects/ckpt/299`（**28 文件 / 9,363,286,152 B**，MD5 清单 sha256 `977eb6e9d52630877c938206ddf883f6b4e3aaa635ea7a7b286155355c219cc2`，与 §4.2 **逐字节一致**）。两种形态均实测：
  - 直接 shim（8401）：`Listening on ws://0.0.0.0:8401`，`GET /healthz` → **200**；
  - **wrapper 形态（8402，与 `qwen_lora_rl` 启动 argv 完全一致：`<SERVER_PY> scripts/serve_pi05.py --config … --checkpoint … --port …`）**：12 s 起服，`/healthz` → **200**，metadata `config_name=pi05_libero_lora_neutral_v5init`、`checkpoint_dir=…/ckpt/299`，日志确认 `Loaded norm stats from /home/yhbcode000/projects/ckpt/299/assets/local/prefmem_neutral`。
  - 资源：加载峰值 GPU **13,721 / 16,376 MiB**（2% util）；探活后已停止服务，GPU 回落到 1,195 MiB（未占用）。
  - 日志：`artifacts/hf_pull_logs/serve_pi05_probe.log`、`serve_pi05_wrapper_probe.log`。
- **SERVER_PY 替换（条件 4）**：新增 `prefmem-libero/local_shims/`（**附加文件，不改 vendor、不改 `scripts/`**，故 §4.4 的 277 文件基线不变）：
  - `serve_pi05_local_config.py`：运行时注册 `pi05_libero_lora_neutral_v5init`（模型配置逐字取 `pi05_libero`，仅 data/asset id → `local/prefmem_neutral`，与 ckpt assets 对应），再调用未修改的 `scripts/serve_pi05.py`；
  - `run_local_a_arm.py`：只重写 `SERVER_PY`/`OPENPI_SRC`/`CLIENT_SRC` 三个常量后委托 `qwen_lora_rl.main()`；自检输出 `SERVER_PY=/home/yhbcode000/projects/openpi/.venv/bin/python`（**未解析符号链接**，保留 venv site-packages）、`OPENPI_SRC=…/openpi/src`、`CLIENT_SRC=…/packages/openpi-client/src`；
  - `serve_py_wrapper.sh`：作为 `SERVER_PY` 可执行替身（忽略占位 argv[1]，转发其余参数到注册 shim）——条件 4 的实际落地形态；
  - `README.md`：调用形态、证据命令、偏差说明。
  - import 自检：`import qwen_lora_rl` → **IMPORT_OK**（此前缺 `peft`，已补装）。
- **已知偏差（须 Lead 知悉）**：本轮**一个 venv 兼任 server + runner**（runner 额外需要 `peft`/`accelerate`，仓内无依赖声明）；更干净的拆分为 `a_arm_runner/.venv`，本轮未拆。serving 不依赖 peft，且探活已在装后复跑通过。
- 结论：条件 1–4 均满足 → **B1b RESOLVED**；FIX 之后的实际 A 臂训练仍受 B6/B7（parity 待裁定）与开训令闸门约束。

### B2 · 训练数据 / episode dumps｜**RESOLVED（2026-11-01 01:53，Local Admin RO-copy）**
- 证据（ALWM `E:\Envs\JitRL_LoRA_Env`，实际落盘）：
  - `prefmem_lora\data\local\`：**21 个 LeRobot 数据集**，合计 `DATA_TOTAL files=350 bytes=6269358674`，`DATA_LISTING_SHA256 = b5da3b9ceddfe4fcd22b9c49e6257ba98290f9755124da9eebda8d3b479f8986`（逐子目录 `files/bytes/listing_sha256` 见 `_b2_inventory.txt:185-207`）。
  - `prefmem_lora\openpi_assets\`：24 个 `norm_stats.json`，**全部 MD5 `03d102a77c0aa343afbf7569c97bbcba`**（与 §4.3 的 299 内副本一致；`_b2_inventory.txt:160-184`）。
  - candidates：`prefmem-libero\runs\online_candidates\`、`prefmem-libero\runs\online_candidates_qwen35\`（逐任务 JSON 的 MD5 见 `_b2_inventory.txt:1-159`，含 A 臂 8 任务的 `3-1_004_1_blue` / `3-1_006_1` / `3-2_032_7` / `3-2_032_8` / `3-2_097_8` / `3-2_097_10` / `3-9_118_1` / `3-10_117_1`）。
  - 目录体积：`prefmem_lora` 5.9 GB。
- 残留（**不属于 B2**，归 B1b）：A 臂运行所需的 `upstream/LIBERO` 代码、openpi venv（`scripts/serve_pi05.py` 宿主）尚未装配。
- 判据：数据与候选集已按 Admin 清单落盘且带清单级 SHA256 → B2 清除。

### B3 · 基座权重｜**RESOLVED（2026-11-01T02:59:47+08:00，HF phase1 `PHASE_DONE`；4B 由 Admin RO-copy）**

#### B3a · `Qwen/Qwen3.5-4B`（Local Admin RO-copy）｜**RESOLVED（复核 2026-11-01T02:45:56+08:00）**
- 路径（双机同字节）：
  - ALWM：`E:\Envs\JitRL_LoRA_Env\models\Qwen3.5-4B`
  - 4090：`D:\Envs\JitRL_LoRA_Env\models\Qwen3.5-4B`
  - 源：`.109:/hhd3/yanghaobo/models/Qwen3.5-4B`
- 规模（两机实测一致）：`files=43`、`bytes=9,342,908,952`（= Admin 声明值；与 `.109` 源一致）
- 时间语义：文件 mtime **保留源时间**（`config.json` 2026-10-29 23:18:15、`shard-00001` 2026-10-29 23:25:32 +0800）——**拷贝日 ≠ 文件日**，勿混淆。
- MD5（**ALWM 本轮独立复算**；4090 侧抽检 `config.json`/`index.json` 一致；RD 抽检 `00001` + `tokenizer.json`）：

| 文件 | MD5 | 校验方 |
|---|---|---|
| `model.safetensors-00001-of-00002.safetensors`（5,329,398,688 B） | `8e5bd9992c08f7d6cbb098b1cc6c9eba` | **ALWM 复算 ✔** + RD |
| `model.safetensors-00002-of-00002.safetensors`（3,990,429,408 B） | `011b980fcfed5178b521f7284ded325c` | **ALWM 复算 ✔** |
| `config.json` | `e570efc2ec15635866fc1a5adfc45cb0` | ALWM ✔ + 4090 ✔ |
| `model.safetensors.index.json` | `14dbb6066c9c64b79da17260cce41d25` | ALWM ✔ + 4090 ✔ |
| `tokenizer.json` | `7b6074791b84e7ce3a5fc12746b82d48` | ALWM ✔ + RD ✔ |
| `tokenizer_config.json` | `53de78b01fb7a555749b2237f1c94110` | ALWM ✔ |
| `vocab.json` | `4e976260d822eb1335417fe34309be3f` | ALWM ✔ |

- 用途：A 臂 `qwen_lora_rl.py --model <本机绝对路径>`（§5.1）。
- 注：`.109/hhd3/yanghaobo/models` 另有 `Qwen3-VL-8B-Instruct`，**未拷贝、未拉取**（MANIFEST 无此需求）。

#### B3b · 其余基座（ALWM **WSL** 缓存 `/home/yhbcode000/.cache/huggingface/hub`）｜**RESOLVED（全部 `EXIT=0`）**
- 日志：`artifacts/hf_pull_logs/wsl_phase1.log`（`RESTART_NONXET 01:51:18` → `PHASE_DONE 2026-11-01T02:59:47+08:00`）；`hf_pull.sh` / `hf download` 进程**已全部退出**。

| repo | EXIT | snapshot ref | 体积 | snapshot 链接 / 断链 |
|---|---|---|---|---|
| `lerobot/pi05-libero` | 0 | `9ba5431256f4…` | 9.9 GB | 9 / **0** |
| `Qwen/Qwen3.5-2B` | 0 | `15852e8c1636…` | 4.5 GB | 13 / **0** |
| `nnh-pbbb/paligemma-3b-pt-224`（tokenizer） | 0 | `ee1ed4a8fbec…` | 22 MB | 11 / **0** |
| `lerobot/pi05_libero_base` | 0 | `5fcf06fc42b6717e22ab885fd381a096574b12c7` | 14 GB | 6 / **0** |

- 完整性判据：4/4 `EXIT=0` + 每个 snapshot 软链**全部可解析（broken=0）** + `pi05_libero_base` 下 `.incomplete=0`。
- 遗留垃圾（**不影响完整性、本轮未清理**，遵守"只改 MANIFEST"范围）：3 个孤儿 `.incomplete` 残留于早前被杀的 xet/非 xet 尝试：
  - `.../models--lerobot--pi05-libero/blobs/4d85aca1….737aa8cc.incomplete`（1,024,000,000 B）
  - `.../models--lerobot--pi05-libero/blobs/4d85aca1….2e86cd28.incomplete`（220,200,960 B）
  - `.../models--Qwen--Qwen3.5-2B/blobs/aa33250c….a6cfce14.incomplete`（268,247,252 B）
  - 清理动作留待下轮（删除前需确认无进程持有；上述 snapshot 均已解析成功，删除安全）。
- 通道要点（复现）：xet 后端会挂死（`1,024,000,000 B` 处 delta=0/30 s）→ 必须 `HF_HUB_DISABLE_XET=1`；WSL 不继承 Windows 代理但直连 `huggingface.co`/`hf-mirror.com` 均可用（~2.6 s）。
- 4090 侧：**无 HF 缓存**（其 01:5x detached 拉取仅装包未落盘；4090 出列，B5）；4090 仅备齐 `models/Qwen3.5-4B` 与 `prefmem_lora` 拷贝。

### B4 · `third_party` 代码填充｜**RESOLVED（2026-11-01T02:45+08:00，范围：仅"代码按 pin 落位"）**
- 结论：本轮 B4 的判据是"能 clone 则 clone 并记 commit"——**两者均已按 gitlink pin 落位并与树内 pin 逐一复核一致**；运行时装配（openpi py3.11 venv、libero 软链、HF 资产）**不属于 B4**，已全部归到 **B1b**。
- 证据：
  - `git ls-tree HEAD:third_party` → **gitlink（mode 160000）**，pin 冻结在 commit 内（引入于 `c76fb6e rlinf fix`）：
    - `third_party/openpi` → `15a9616a00943ada6c20a0f158e3adb39df2ccac`
    - `third_party/libero-pro` → `eafdb809426b13153aa1e4c42d6601844217dfec`
  - `.gitmodules` **全历史不存在**、`.git/config` 无 `[submodule]` 段 → 上游 URL 在树内**无声明**；URL 为**按 SHA 反查推断**（GitHub `/repos/<repo>/commits/<sha>` = 200）：
    - openpi → `https://github.com/Physical-Intelligence/openpi`
    - libero-pro → `https://github.com/Zxy-MLlab/LIBERO-PRO`（来源：arXiv 2510.03827 页面链接；另 5 个候选仓 422/404 已排除）
  - 落地复核：`git -C third_party/openpi rev-parse HEAD` = `15a9616a…`（27 MB）、`git -C third_party/libero-pro rev-parse HEAD` = `eafdb809…`（990 MB，01:37 完成）→ **两目录 HEAD 与 gitlink 一致**，故 `git status --porcelain` 不出现子模块条目（`.gitignore` 另已覆盖 `.venv/` 与 `artifacts/`）。
- 未随 B4 清除（全部转 B1b）：`third_party/openpi/.venv`（py3.11+torch2.7.1）、`third_party/libero-pro/libero` 软链与 `~/.libero/config.yaml`、`zhouxueyang/LIBERO-Pro` 数据集 bddl/init。
- 警示：B4 RESOLVED **不等于** `rlinf` / `libero_pro` 后端可用。

### B5 · 平台与执行范围｜**RESOLVED（2026-11-01 01:55，Lead steer：本轮 ALWM-only）**
- 裁定（Lead 转述）：**本轮及后续本科学问的执行范围限定 ALWM18R2**；4090 机不参与执行，仅在 Lead/Admin 明确另行下发时启用。
- 证据：README:22-26 要求 Ubuntu 24.04/Linux CUDA 且禁止从 `/mnt/c`、`/mnt/d` 跑大实验；ALWM 有 `Ubuntu-24.04`（Running, WSL2）+ ext4 checkout `~/projects/JitPi05-jitrl-vs-lora-acc` @ `5472f1f`，`nvidia-smi -L` 在 WSL 内可见 `RTX 4090 Laptop`；4090 仅 `docker-desktop`（无 Ubuntu 发行版）——该缺陷因 4090 出列而不再阻塞。
- 仍适用的约束：Jit 臂只能在 WSL ext4 内运行（不得 `/mnt/e`）；A 臂同理（若在 ALWM 执行需同样避免 `/mnt/*`）。
- 遗留（记 B3/B6）：4090 侧的权重/环境不属本轮范围。

### B6 · 两臂模型/执行器/学习语义不对齐｜**RESOLVED（2026-11-01T~07:36+08:00，Haobo 裁定 option (1) TRUE ALIGNMENT REPLACE 吸收）**
- 裁定把三项从"待协商差异"变成"强制固定"：**同 backbone（Qwen3.5-4B）**、**同执行器（299 线）**、**B 必须是权重训练**（并明令禁止测试期 logit 调制 / 2B 对照 / 叠加 LoRA）。
- 本文件据此重写：§1 声明裁定与禁止项；§6 删去旧 "test-time + 2B" 设计并改为训练-Jit 缺口盘点；§7 把 backbone/执行器/任务/种子/预算/指标锁成两臂共享合同（唯一变量=更新规则）。
- 旧 B6a/B6b/B6c 的"不对齐"描述已随设计作废而失效，不再构成 blocker；**B 臂的实现缺口**改记为 **B8**（同一根因，归到正确的 blocker）。
- 证据：本文件 §1/§6/§7；裁定时间 2026-11-01 ~07:36 Asia/Shanghai（Lead Explore 转述 Haobo）。

### B7 · 任务集零交集｜**RESOLVED（2026-11-01T~07:36+08:00，同上裁定）**
- 裁定要求"同任务集"，故任务集不再是可选协商项：**B 臂必须使用 A 臂既有 8 个 id**（§6.4），不得改用 stock `libero10/libero90/libero_pro` 面板。
- 该约束已写入 §7 固定合同表（第一行）；任何偏离视为违约，不再需要"二选一"裁定。
- 说明：B 臂**尚未实现**，所以"B 侧任务注册"这一步尚未发生——这是实现缺口（B8），不是对齐缺口（B7）。

### B8 · 【原 07:36 版】"B 必须是权重训练"｜**RESOLVED / SUPERSEDED（2026-11-01T~07:44+08:00）**
- 07:44 裁定把 B 臂改为 **JitRL 测试期**等价替换，明确"NOT a new weight-train entry and does NOT reopen B8 weight-train demand"。
- 因此原 B8（"权重训练目标在仓内/论文中无定义"）**不再成立/不再需要清除**；相关证据（JitPi05 0 训练原语、`JITRL.pdf` gradient-free）留档于 §14，作历史依据。

### B9 · B 臂执行链｜**RESOLVED（2026-11-01T09:21:31+08:00，8/8 mission 跑通）**
- **B9b-1 mission 包：RESOLVED（2026-11-01T08:03+08:00，Local Admin RO-copy + 本地复核）**
  - `E:\Envs\JitRL_LoRA_Env\prefmem-libero\upstream\LIBERO`：`mission_info.yaml` **8/8**；
    missions `files=72 bytes=547,303,404`、assets `files=917 bytes=721,608,920` → 合计 **989 / 1,268,912,324**，与 Admin 声明**逐项相等**。
  - 8 个 mission 目录（相对 `prefmem-libero/`）：`upstream/LIBERO/dataset/series_3/perm_3_1/libero_004_1_blue|libero_006_1`、`perm_3_2/libero_032_7|032_8|097_8|097_10`、`perm_3_9/libero_118_1`、`perm_3_10/libero_117_1`。
  - `mission_info.yaml` MD5（8 条已 pin）：`6f1b979d…(004_1_blue)`、`f4277e88…(006_1)`、`f2ef2f7e…(032_7)`、`eba9e5a5…(032_8)`、`cfe20ddf…(097_8)`、`8c2f18ca…(097_10，=源 .bak)`、`0c086d93…(118_1)`、`ba247a5b…(117_1)`。
  - ⚠ 清单摘要**不可交叉核验**：本文件按两种公式自算 missions `c16f1165…(relpath+size)` / `aff1bf8f…(md5+relpath)`、assets `413c15ef…` / `996ff351…`，均 ≠ Admin 的 `98bad794…` / `e6bfd6a7…`（公式未公布）→ 只声明**文件数+字节完全相等**，摘要公式差异待 Admin 配方。
  - ⚠ Admin 备注：07:5x 有并发写者改写 `perm_3_2/libero_097_10/mission_info.yaml`（984 B），已从 `.bak` 恢复为 952 B（`8c2f18ca…`）；`.bak` 仍在 `097_8`、`117_1` 旁（保留）。
- **B9b-2 执行器代码：RESOLVED（只读自 .109 补拉）**：`PolicyEval/`、`Preference/`、`Utility/`、`configs/`、`libero/libero/{benchmark,utils,init_files}`、`libero/{configs,lifelong}` 全部到位（树内 `files=1564 bytes=1,295,266,713`）。
- **B9b-3 selector→decision：RESOLVED（真 mission）**：`3-1_006_1` →
  `plan_jitrl_selector.py`（Q/V/A/β）输出 `artifacts/hf_pull_logs/b9_decision_006.json`（`selected=pref`，
  plan = `pick alphabet_soup_1 → put basket_1 → pick blue_cream_cheese_1 → put basket_1`）。
- **B9b-4 299 服务器半：RESOLVED**：`/healthz` **200**（12–18 s 起服）+ ws 探针 `actions[10,7]`、`ok=true`（§14）。
- **结论（09:21:31）**：`policy_eval done: **8 ok, 0 failed**`，`CLIENT_RC=0`；**8/8 mission** 各产出 `rollout_summary.yaml`。
  - 证据目录：`artifacts/hf_pull_logs/b9_run8_summaries/<mission_id>.yaml`（8 个文件）；
  - 首帧真实 obs 键（shim 日志）：`['observation/eef_ori_mat','observation/image','observation/state','observation/wrist_image','prompt']`，形状 `(224,224,3)/(224,224,3)/(8,)/(3,3)`；
  - 服务端 `/healthz` 200；`infer` 正常返回动作块；
  - **单次 trial 的 SR（非科学结论，仅链路验证）**：`3-2_032_7=1.0`、`3-2_032_8=1.0`，其余 6 个 `0.0`。
  - **最小 infra patch（记录 diff 与理由）**：`libero/libero/benchmark/__init__.py::grab_language_from_filename` —— stock suite 的 bddl 文件在本 fork 中不存在（本地 361 = .109 361），导入期 `BDDLUtils.get_problem_info()` 会 `FileNotFoundError`；改为 `try/except FileNotFoundError → 由文件名派生 language`。仅影响 stock suite 的 language 字符串，**不影响** prefmem mission 的 bddl/env/policy。
  - **复现运行时规格**：`~/projects/libero_rt311/.venv`（**py3.11**）+ `robosuite==1.4.1` + `mujoco==2.3.7` + `numpy==1.26.4`（`mujoco 3.x` 会触发 free-joint 断言）；`TaskGen/TaskEval/tools/templates/visualizer_v2/scripts` 与 `libero/libero/{benchmark,utils}` 均已补齐。
  - **正式跑前的口径提醒**：本次 dry-run 用 `seed 7 / 1 trial`（`run_plan_with_vla.py` 的 runfile 固定值）；§7 合同要求 **seed 0**（用直连 runfile 可设），正式评测为每 task **5 episodes**。
- **历史记录（B9 修复过程）**：
  - **根因已定位并修复（08:23）**：WSL 树 `libero/` 一直是 **vanilla clone**（此前仅补 PolicyEval/Preference/Utility/configs），registry 只有 stock 50 项；用 E: fork 覆盖后 `objects/__init__.py` MD5 `632ead5e…`→`00276ed3…`（= .109），`OBJECTS_DICT` **991 项**、`green_butter=True`、`get_object_fn("green_butter")→GreenButter`。
  - **env 已建**：08:24 `run_plan_with_vla.py` 完成场景构建并进入 rollout（PolicyEval 打印 prompt、连上 299）。
  - **协议缝已定位（08:36，shim 已加观测钩子）**：服务端收到的**第一条**消息是 `{'__end__': True}`（=`PolicyEval.shutdown_server()` 的结束哨兵），openpi server 无哨兵处理 → 被当作 obs 送去 infer → 报 `KeyError: 'observation/image'`。**并非 obs schema 不兼容**（手工 4/5 键探针均 OK）。
  - **shim 已修**：`serve_pi05_local_config.py` 现对 `__end__` 短路返回 `{'__end__': True}`（不再 infer）；同时保留 `infer#N keys=...` 首次观测日志与 alias 归一化。
  - **真正的剩余阻塞（08:40 直接跑 `run_client.sh` 抓到）**：episode 在首个 obs 之前失败 ——
    `robosuite/utils/binding_utils.py:521` `assert joint_type in (HINGE, SLIDE)` → **`AssertionError`**（env reset 阶段）。
    根因 = **运行时栈版本不兼容**：`mujoco 3.17.0` + `robosuite 1.4.1`（fork 需要 1.4.x 的 `single_arm_env`）。尝试
    `mujoco==2.3.7` 失败：`error: Failed to build mujoco==2.3.7 ... RuntimeError: MUJOCO_PATH environment variable is not set`
    （py3.12 无 cp312 wheel，需源码构建）。旁证：.109 fork 的 `__pycache__` 为 **cpython-310** → 其运行环境是 **py3.10**，
    而我们的 `~/projects/libero_rt/.venv` 是 py3.12。
  - **bddl 根布局待确认**：`libero/libero/bddl_files/` 本地仅 7 个**子目录**（`libero_10/90/goal/object/pbench/pbench_eval/spatial`），
    mission 声明的 `LIVING_ROOM_SCENE1_….bddl` 是 flat 名称；`PolicyEval/eval.py` 用 `get_libero_path("bddl_files")/<bddl_file>` 组合 →
    需确认该 bddl 实际位于哪个子目录（或 `.109` 的 `LIBERO_CONFIG_PATH` 布局）。
  - **① 运行时栈：已解决（09:00）**：`~/projects/libero_rt311/.venv`（**py3.11**）+ `robosuite==1.4.1` + `mujoco==2.3.7` + `numpy<2` → `STACK 2.3.7 / 1.26.4 / 1.4.1`；mission bddl（`bddl_files/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl`）**reset 成功**：`ENV_OK`。
  - **② 代码树完整化**：补 `TaskGen/TaskEval/tools/templates/visualizer_v2/scripts/setup.py/requirements.txt` 与 `libero/libero/{benchmark,utils}`；`import TaskGen` → **TASKGEN_OK**。
  - **③ 当前唯一阻塞（09:07）**：客户端 `import libero.libero.utils` → 触发 `libero.libero.benchmark` 的**导入期 bddl 扫描** →
    `FileNotFoundError: …/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_1.bddl`；
    实测该文件在 **.109 同样不存在**（本地 `*.bddl` **361 = .109 361**）→ 该 fork **不发 stock bddl 全集**，eager 扫描在本树必失败。
  - **下一步**：让客户端**不导入** `libero.libero.benchmark`（shim/runner 侧跳过或确认 eval 不需要），或补 stock LIBERO 官方 bddl 全集；均不改 vendor 逻辑。
  - 历史失败（保留）：
  1. 用 JitPi05 venv（含 stock `libero`）跑 `run_plan_with_vla.py` → `PolicyEval/eval.py` 在 `libero_living_room_tabletop_manipulation.py:59`
     抛 **`KeyError: 'green_butter'`**：`import libero` 被 site-packages 的 **stock libero 遮蔽**（`OBJECTS_DICT` 无自定义物体）。
  2. 改用**干净 venv**（`~/projects/libero_rt/.venv`，无 `libero` 安装）+ `PYTHONPATH=<fork>`：`libero.__file__` 已为 **None（namespace=fork）**，
     但 `green_butter in OBJECTS_DICT` 仍为 **False**（fork 的 `hope_objects.py` 确有 `@register_object class GreenButter` → 注册键应为 `green_butter`；
     说明还有一处 fork 启动/导入顺序未复现）；依赖侧已装 `robosuite==1.4.1`（含 `single_arm_env.py`）、`mujoco`、`bddl`、`easydict`、`scipy`、`scikit-image`、`cloudpickle`、`gym==0.26.2`、`imageio`、`msgpack`、`websockets`。
  3. 参考运行时 `/nvme-ssd/yanghaobo/pi05g_ws/libero_venv` 在 .109 **不存在**（无法取其 freeze 作 pin）。
- **B9c suite 注册：RESOLVED（2026-11-01T08:22+08:00，数据注册）**
  - 原始：`single-pref` 只含 8 个中的 3 个；`3-2_097_8/097_10` 仅在 `liberoscale*`；`3-1_004_1_blue`、`3-2_032_7`、`3-2_032_8` 无任何 suite。
  - 处理：向 `configs/task_suites.yaml#single-pref.missions` 追加 5 个缺失 id（带 `# [local-wiring 2026-11-01] A-arm 8-task alignment` 注释），40 → **45**，现含全部 8 个；未改代码、未改既有条目。
  - 注：`eval.py:651` 的 `verify_suite` 失败仅 warning；`VLA_data` 缺失时 `check_suite` 跳过跨界检查。
- 清除条件：env-half 跑通（`green_butter` 注册 + 环境 reset + 一步动作）并产出真 mission 的 `rollout_summary.yaml`；B9c 由 Lead/Admin 决定注册或换 suite。

### B10 · 参考偏好对齐入口依赖｜**RESOLVED（2026-11-01T08:07:40+08:00，cold-start 口径）**
- `plan_jitrl_vlm_text_selector.py:15-20` 需要 `--mission-dir/--candidates/--memory/--image` 四件输入；`vlm_text_key.load_key()` 读 `DEEPSEEK_API_KEY` 或 `/tmp/deepseek_key`（`vlm_text_key.py:18-23`）→ key 见下方 07:57 条（已落盘可注入）；**memory/image 落盘件仍为 0**。
- JitPi05 侧无替代：`grep -rn -i "peft|LoraConfig|load_adapter|PeftModel|prefmem|preference_alignment" src scripts docs plans reports` → **全 0**（只有 `pairwise` 18 次出现在 issue 文档文字里）。
- **key 已可注入（2026-11-01T07:57+08:00）**：Windows 用户环境存在 `DEEPSEEK_API_KEY`（长度仅记录为 35，**未打印值**）；已落盘为受限文件 `E:\Envs\JitRL_LoRA_Env\.secrets\deepseek_key`（36 B，ACL 已 `/inheritance:r`，仅 `ALWM18R2\Admin:(F)`）。供 WSL 使用路径：`DEEPSEEK_API_KEY_FILE=/mnt/e/Envs/JitRL_LoRA_Env/.secrets/deepseek_key`（`vlm_text_key.py:20` 读取该 env）。
- **输入现状（08:07 后）**：候选 JSON **20 个**；**图像 ✓**（mission 自带 `screenshot/agentview_image_1.png`）；**memory = cold-start 空**（`TextMemory().save`，README 定义的初始态，非合成数据）。真实 memory 只在 rollout 后累积。
- **调用坑（已记录）**：`plan_jitrl_vlm_text_selector.py` 的 `--candidates` 是相对路径时须以 **prefmem-libero 根**为 cwd 运行（在 `scripts/` 下运行会解析成 `scripts/runs/...` → FileNotFoundError）。
- 需要的落盘输入：每 mission 的 `--memory`（`TextMemory` JSON）与 `--image`（首帧 PNG）；当前一件都没有。
- **已实测（08:07:40）**：`plan_jitrl_vlm_text_selector.py` 用**真 mission 截图**（`.../libero_006_1/screenshot/agentview_image_1.png`）+ **空 memory（cold start，`TextMemory().save`）** 跑通：DeepSeek 返回真实场景 JSON（basket/can/green box/robot arm…），Q/V/advantage/probabilities 全量输出，`SELECTOR_EXIT=0`，decision → `artifacts/hf_pull_logs/b10_decision_dryrun.json`。
- **协议更新（superseded by `ADDENDUM3_v2_lead_1101.md` §(ii)，MD5 `64c31a8507a87b217217c095e7415e1d`）**：B 臂**不再是"每任务冷启动"**。
  - **一份 memory 贯穿全 run，跨任务累积**；**任务顺序与 A 臂完全一致**（8 task 原序）。
  - **每任务**：**12 个 adapt episodes**（记忆累积）→ **5 个计分 episodes（记忆冻结）** → 携带该记忆进入下一任务。
  - **每任务结束**保存 memory 快照 + **MD5**（对齐 A 臂每任务 LoRA 快照 + MD5）。
  - 本文件此前"每任务独立空 memory（README:88 cold start）"的表述**对 B 臂 v2 作废**；selector dry-run 的 cold-start 数字仅作管道证据保留。
- **历史口径（仅 dry-run 证据，不再适用于正式 B 臂）**：空 memory ⇒ `Q=0.5`、`selected_index=0`；真实运行由 rollout 逐步写入 memory（`plan-level memory` 仍需按 §B9 的 env-half 落地后才能累积）。
- 未打印任何密钥值；key 文件路径见上（ACL 已收紧）。

### 非阻塞（记录，不改变 STATUS 判定路径）
- `pi05g_ws` 按约定跳过（Haobo 指示）。
- 4090 机一处 **Unicode PNG 解压失败**（libero-pro）——与 B4 的空目录/`libero_pro` 面板不可用一致；不作为独立 blocker，若启用 `libero_pro` 面板须先修（见 B4）。

---

## 12. GATE


### 12.1 Pre-launch 条件（READY ≠ 已授权开跑）

1. **FC2**：`docs/papers/CycleVLA.pdf` 的既有 `M` 改动（MD5 `96dc4ba2…`）需 Lead 书面接受为基线；
2. **seed=0**：dry-run 用的是 `run_plan_with_vla.py` runfile 默认 seed 7；正式跑须用直连 runfile 固定 `seed 0`（§7 合同）；
3. **episodes**：每 task 5 episodes（dry-run 为 1 trial）；
4. **FC13**：A 臂历史 `results/qwen35_lora_rl8*.json` 的 `updates=0` 已记录，作基线前需 A 臂负责人确认；
5. **开训令**：`LEAD_EXPLORE_START_ORDER`（本文件不构成授权）。
- **本文件不授权任何开训。** 执行方（Exp）只有在同时满足以下全部条件后才能启动任何一侧：
  1. 本文件 `STATUS` 被改为 `READY`，且 §11 每条 blocker 标 `RESOLVED` 并附证据（命令输出 / MD5 / 时间戳）；
  2. 拿到 **Lead Explore 开训令**（唯一授权来源），令中须包含：允许的主机、允许的 GPU、run_id、两臂命令全文、预算上限；
  3. §10 的 FC1–FC12 全部实测通过，输出写入 `PROVENANCE.md`。
- 未拿到开训令前的允许动作：只读核验、安装依赖（B1）、下载权重/数据（B2/B3）、拉子模块（B4）、建 WSL（B5）——任何一项做完都要回来更新本文件证据位。
- 禁止动作：开训、造数、改训练代码、push、触碰 `full74v3`/Dig 闸门。

---

## 13. 复现命令（重算本文件所有指纹）

```bash
# 1) checkpoint 文件级 MD5（28 行，对照 §4.2）
cd <root>/checkpoints/pi05_libero_lora_neutral_v5init/pi05_neutral_lora_v4_v5init/299
find . -type f -exec md5sum {} + | sort -k2 | tee /tmp/ckpt299.md5
sha256sum /tmp/ckpt299.md5                     # 期望 977eb6e9d52630877c938206ddf883f6b4e3aaa635ea7a7b286155355c219cc2
find . -type f -printf '%p\t%s\n' | sort | sha256sum   # 期望 a9ee6806bb48b56662c7fb3c8255d1f974f9624336bd5cb6a4888a6b63819ca5
find . -type f -printf '%s\n' | awk '{s+=$1} END {print s}'  # 期望 9363286152

# 2) 4090 侧（Windows PowerShell，等价 Get-FileHash）
#    $p='D:\Envs\JitRL_LoRA_Env\checkpoints\...\299'
#    Get-ChildItem -LiteralPath $p -Recurse -File | Sort-Object FullName | %{ "$((Get-FileHash -Algorithm MD5 -LiteralPath $_.FullName).Hash.ToLower())  ./" + $_.FullName.Substring($p.Length).TrimStart('\').Replace('\','/') }

# 3) prefmem scripts 双机对等（277 个文件）
cd <root>/prefmem-libero && find scripts -maxdepth 1 -type f -exec md5sum {} + | sort -k2 | sha256sum
#    期望 5d1a35df0320983081b89c6e1a3e9d61e397fb22d481b26e3b058eede017cc98（两机实测相等）
#    4090 侧等价（PowerShell；输出写成 UTF-8 后 scp 回本地再比对，注意 CRLF→LF 归一）：
#    $p='D:\Envs\JitRL_LoRA_Env\prefmem-libero\scripts'
#    Get-ChildItem -LiteralPath $p -File | %{ "{0}  scripts/{1}" -f (Get-FileHash -Algorithm MD5 -LiteralPath $_.FullName).Hash.ToLower(), $_.Name } | Sort-Object

# 4) 环境/平台
git -C <root>/JitPi05-jitrl-vs-lora-acc rev-parse HEAD && git -C <root>/JitPi05-jitrl-vs-lora-acc status --porcelain
ls -d <root>/JitPi05-jitrl-vs-lora-acc/.venv || echo NO_VENV
uv sync --frozen && uv run jitpi05-doctor
wsl -l -v
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
```

---

## 14. 核验日志（本轮实测时间线，2026-11-01）

| 时间（+08:00） | 动作 | 结果 |
|---|---|---|
| 01:13 | ALWM 299 全目录 MD5（9.36 GB，28 文件） | 完成，17.9 s |
| 01:14 | 4090 `299` 全目录 MD5（PowerShell） | **28/28 与 ALWM 相同**；`D:` free 1326.0 GB |
| 01:15 | 两机 `nvidia-smi` 快照 | ALWM 1318/16376 MiB；4090 12588/24564 MiB |
| 01:15 | 两机 `.venv` / WSL / HF cache 探测 | 均无 `.venv`；4090 无 Ubuntu WSL、无 HF cache |
| 01:16 | 两机 `third_party` / `.gitmodules` | 空目录 + 无 submodule 映射 |
| 01:18 | `openpi_assets` 副本 MD5（两机） | `03d102a77c0aa343afbf7569c97bbcba`，与 299 内副本一致 |
| 01:19 | 脏文件 `docs/papers/CycleVLA.pdf`（两机） | `96dc4ba22b674a74d12b2a3fa0672a15`，13,407,989 B，同为 `M` |
| 01:21 | `prefmem-libero/scripts` 双机逐文件 MD5（277 文件） | **全等**；清单 SHA-256 `5d1a35df0320983081b89c6e1a3e9d61e397fb22d481b26e3b058eede017cc98` |
| 01:22 | 4090 主仓 `JitPi05` 分支 | `eliminate` @ `5472f1f`；主仓 26 条既有未提交改动（与本 manifest 无关） |
| 01:21 | ALWM Windows `uv sync --frozen` | **退出码 2**：`triton==3.6.0` 无 `win_amd64` wheel（`.venv` 建成后失败） |
| 01:26 | 4090 Windows `uv sync --frozen`（uv 0.13.5 / py3.12.10） | **同一错误、同一退出码**；另发现 `WinGet\Links\uv.exe` 为非交互 ssh 下不可执行的符号链接 |
| 01:22 | `git ls-tree HEAD:third_party` | 两目录为 **gitlink 160000**：openpi `15a9616a…`、libero-pro `eafdb809…`；`.gitmodules` 全历史缺失 |
| 01:23 | GitHub API pin 反查 | `Physical-Intelligence/openpi` 200；`Zxy-MLlab/LIBERO-PRO` 200（arXiv 2510.03827 页面链接）；另 5 个候选仓 422/404 |
| 01:23 | openpi clone @ pin | 完成，`HEAD=15a9616a`，27 MB（git status 不显示差异，因与 gitlink 一致） |
| 01:24-01:37 | libero-pro clone（355 MB 仓） | 首轮被 300 s 工具上限中断（75 MB）；`fetch --depth 1 <sha>` 脱离续传 → **01:37 完成**，`HEAD=eafdb809`（与 gitlink 一致），990 MB |
| 01:37 | 子模块 pin 复核 | `third_party/openpi=15a9616a`、`third_party/libero-pro=eafdb809`，与 `git ls-tree HEAD:third_party` 完全一致；`git status` 仍只有 MANIFEST + 既有 PDF |
| 01:22-01:31 | WSL `uv sync --frozen`（`~/projects/JitPi05-jitrl-vs-lora-acc`） | 首次被 300 s 工具上限中断；改 detached 续跑，日志 `/tmp/sync.log`（WSL 内） |
| 01:29 | WSL HF 拉取（detached） | `lerobot/pi05-libero` → `Qwen3.5-2B` → tokenizer → `Qwen3.5-4B` → `pi05_libero_base`，日志 `artifacts/hf_pull_logs/wsl_phase1.log` |
| 01:47 | 发现 xet 通道挂死 | `.incomplete` 停在 `1,024,000,000 B`、30 s delta=0 → 改 `HF_HUB_DISABLE_XET=1` |
| 01:53 | Local Admin RO-copy 落盘（ALWM） | `prefmem_lora/` 5.9 GB（data 21 数据集 / 350 文件 / 6,269,358,674 B / `DATA_LISTING_SHA256 b5da3b9c…`）、`prefmem-libero/runs/online_candidates{,_qwen35}`、`_b2_inventory.txt`（208 行，含 candidates/assets/data 三组 MD5） |
| 01:55 | Lead steer：4B 不自行拉取 | 等 Admin 的 `.109:/hhd3/.../Qwen3.5-4B` RO-copy（`models/` 仍 0 B）；WSL 拉取列表移除 4B |
| 01:55 | Lead steer：执行范围 ALWM-only | B5 按范围裁定 RESOLVED；4090 出列（其 detached 拉取仅备料） |
| 01:57 | WSL HF 重启（无 4B） | `RESTART_NO4B`：`pi05-libero → 2B → tokenizer → pi05_libero_base`；01:58 时 `pi05-libero` 1.3 GB、`2B` 266 MB |
| 02:03:09 | WSL `uv sync --frozen` 完成 | **退出码 0**；`.venv` 6.9 GB / 228 包 / freeze SHA-256 `ea309848…`；`uv.lock` 无漂移（FC5） |
| 02:1x | WSL `uv run jitpi05-doctor` | **退出码 0**：`torch 2.11.0+cu130`、`cuda_runtime 13.0`、`RTX 4090 Laptop`、`compute 8.9`、`bf16 true`、`fla_backend cuda` → B1 RESOLVED |
| 02:1x | Local Admin 4B RO-copy 中间态（**已被 02:2x 行取代**） | `models/Qwen3.5-4B` 4.3 GB（拷贝进行中；最终以 02:2x 复核为准） |
| 02:1x | WSL HF 进度 | `models--lerobot--pi05-libero` **4.8 GB**（增长中）；`Qwen3.5-2B` 266 MB；`HF_RUNNING` |
| 02:2x | 4B 双机落盘复核（RD 抽检 ALWM 通过） | ALWM 与 4090 均 `files=43`、`bytes=9,342,908,952`；shard-00001 `8e5bd999…`（ALWM 本轮独立复算 ✔，RD ✔）、shard-00002 `011b980f…`（✔）、`config.json e570efc2…`（双机 ✔）、`index.json 14dbb606…`（双机 ✔）、`tokenizer.json 7b607479…`（ALWM ✔ + RD ✔）、`tokenizer_config 53de78b0…`、`vocab 4e976260…`；文件 mtime 保留源时间（2026-10-29 23:18/23:25） |
| 02:45 | ALWM-WSL HF 拉取完成项 | `lerobot/pi05-libero` **EXIT=0**（9.9 GB，ref `9ba5431256f4`）；`Qwen/Qwen3.5-2B` **EXIT=0**（4.5 GB，ref `15852e8c1636`）；`nnh-pbbb/paligemma-3b-pt-224` **EXIT=0**（22 MB，ref `ee1ed4a8fbec`） |
| 02:45 | B4 范围裁定 | 两 gitlink 均按 SHA 落位且 HEAD 一致 → B4 RESOLVED（仅代码填充）；运行时装配转 B1b |
| 02:45 | 剩项进行中 | `lerobot/pi05_libero_base`：5.9 GB，in-flight `.incomplete` ≈ 6.26 GB（ref `5fcf06fc42b6`）；`pgrep` 有活进程 |
| 02:59:47 | **HF phase1 `PHASE_DONE`** | 4/4 repo `EXIT=0`：`pi05-libero`（9.9 GB，ref `9ba5431256f4`）、`Qwen3.5-2B`（4.5 GB，ref `15852e8c1636`）、`paligemma`（22 MB，ref `ee1ed4a8fbec`）、`pi05_libero_base`（**14 GB**，ref `5fcf06fc42b6717e22ab885fd381a096574b12c7`）；各 repo snapshot 断链数 = 0；`pi05_libero_base` `.incomplete=0`；hf 进程全部退出（omp/wrapper 未动） |
| 02:59:47+ | 残留垃圾登记（未清理） | 3 个孤儿 `.incomplete`：pi05-libero 1,024,000,000 B + 220,200,960 B；2B 268,247,252 B → 均不影响 snapshot 完整性 |
| 02:59:47+ | **B3 → RESOLVED** | 4B（B3a，双机 43 文件/9,342,908,952 B）+ 其余 4 repo（B3b，全 `EXIT=0`） |
| 03:08:53 | B1b 装配启动（WSL ext4） | `prefmem-libero` 副本 4.4 MB；LIBERO clone；openpi clone @ `15a9616a` |
| 03:12:14 | 299 复制到 ext4 | `~/projects/ckpt/299`：**28 文件 / 9,363,286,152 B**，MD5 清单 sha256 `977eb6e9…`（与 §4.2 逐字节一致） |
| 03:1x | LIBERO 上游落位 | `~/projects/prefmem-libero/upstream/LIBERO` HEAD `8f1084e3132a39270c3a13ebe37270a43ece2a01`（**仓内无 pin，provenance UNKNOWN**） |
| 03:17:45 | openpi venv 完成 | `uv venv --python 3.11` + `uv sync --frozen` → **EXIT=0**；7.7 GB / 242 包；freeze `55b1afcb…`；`torch 2.7.1`/`transformers 4.53.2`/`jax 0.5.3` |
| 03:20:08 | 直接 shim 探活 | `Listening on ws://0.0.0.0:8401`，`/healthz` **200**；norm stats 自 299 载入；GPU 峰值 13,721/16,376 MiB |
| 03:21:18 | wrapper 形态探活（= `qwen_lora_rl` 启动 argv） | 12 s 起服 `ws://0.0.0.0:8402`，`/healthz` **200**；随后停服，GPU 回落 1,195 MiB |
| 03:2x | `local_shims/` 落地 | `serve_pi05_local_config.py` / `run_local_a_arm.py` / `serve_py_wrapper.sh` / `README.md`；`import qwen_lora_rl` → IMPORT_OK；shim 自检 `SERVER_PY=…/openpi/.venv/bin/python`（未解析 symlink） |
| 03:2x | **B1b → RESOLVED** | 条件 1–4 全满足；偏差：单 venv 兼任 server+runner（+`peft 0.21.2`/`accelerate 1.15.0`），已记入 B1b |
| 07:36 | **裁定吸收（option 1 TRUE ALIGNMENT REPLACE）** | 基线=grounded LoRA 299 线；同任务集+同 backbone；B=**训练 Jit（权重训练）**；禁止测试期 logit 调制 / 2B 对照 / 叠加 LoRA |
| 07:4x | B 臂权重训练入口盘点 | `pyproject.toml:27-32` 无 train；JitPi05 `src`+`scripts` 对 `.backward(`/`train_step`/`optax`/`torch.optim` **0 命中**；prefmem 权重更新脚本仅 3 个且全为 LoRA；`plan_jitrl_*` 全为测试期 selector |
| 07:4x | 论文口径核对（`docs/papers/JITRL.pdf`, 842,688 B） | 标题 "…Without Gradient Updates"；摘要 "a training-free framework"；Fig.1 "(z′ = z + βA) … without gradient updates"；方法段 "Instead of updating model parameters θ …" → **gradient-free 定义明确** |
| 07:4x | **B6、B7 → RESOLVED** | 裁定把 backbone/执行器/任务集固定为两臂共享合同（§7）；旧 "test-time + 2B" 设计作废（§1 留一行审计痕迹） |
| 07:4x | **B8 开立** | 裁定要求的"训练 Jit"无 loss/目标定义、无实现；实现即新方法 → 需 Lead 给规格或修订裁定；**STATUS 保持 BLOCKED** |
| 07:44 | **裁定更新（覆盖 07:36）** | B 臂改为 **JitRL 测试期**等价替换（`z'=z+β·A`）；"权重训练"要求撤销 → 原 B8 superseded |
| 07:45 | addendum | B 臂须点名「接地 LoRA 后 JitPi 偏好对齐」参考路径 |
| 07:5x | 参考路径定位（prefmem） | `plan_jitrl_{selector,memory_selector,vlm_selector,vlm_text_selector}.py` + `controlled_loop_109.py`；`plan_jitrl_selector.py:33-44` 需 `--mission-dir`；`vlm_text_key.py:18-23` 需 DeepSeek key |
| 07:5x | 现状审计（原文） | `results/plan_level_jitrl_A_real.md`：`VLA executor = ⚠️ heuristic MotionGen, not π0.5`；Next #1 = *replace heuristic executor with grounded LoRA π0.5 …*（指旧 heuristic 路线）→ 后经 07:51 复核：新 hop `run_plan_with_vla.py` 存在 |
| 07:5x | mission 输入核查 | `find . -name mission_info.yaml` = **0**；8 个 mission id 无本地 mission 包 |
| 07:5x | JitPi05 侧偏好对齐核查 | `grep -rn -i "peft|LoraConfig|load_adapter|PeftModel|prefmem|preference_alignment" src scripts docs plans reports` → **全 0**（`pairwise` 18 次仅在 issue 文案中） |
| 07:5x | 299 可加载性实测 | `pi05_libero` asset_id → **LOAD_FAIL**（缺 `assets/physical-intelligence/libero/norm_stats.json`）；`local/prefmem_neutral` → **LOAD_OK**(2) → 299 需注册 config（已在 `local_shims` 实现） |
| 07:5x | **B8 → RESOLVED(superseded)；B9/B10 开立** | STATUS 保持 **BLOCKED**（B 臂不可运行） |
| 07:49 | mission 包全盘搜索 | `E:\Envs` `find -name mission_info.yaml` → **count=0**；WSL `~/projects` → **count=0**；`green_cream_cheese` 仅出现在 candidates JSON（生成器词表，非 mission 包） |
| 07:50 | 对照叙事钉死（07:50 steer） | §7.1 + frontmatter：论文主张=可替换 RL；本线对照=同 grounded LoRA 设定换更新规则；A=LoRA RL(4B+299)；B=post-LoRA JitPi 偏好对齐(z'=z+βA) |
| 07:51 | hop 代码定位（更正 B9a） | `run_plan_with_vla.py --mission-dir --mission-id --decision-json --port --config --checkpoint --prompt`，由 `controlled_loop_109.py:348-354` 调用；hard-require `mission_info.yaml`（:61） |
| 07:52:18-38 | hop 服务器半干跑 | `local_shims/jit_hop_probe.py` + 假 decision → ws 299：metadata 正确、`actions [10,7]`、`ok=true`、`connect 0.005s`、`infer 16.128s`（首调 JIT）；`/healthz` 200；GPU 峰值 13,749/16,376 → 停服 1,193 MiB |
| 07:52 | B10 依赖实测（WSL 内；**已被 07:57 行取代**） | WSL `/tmp/deepseek_key` NO_KEY_FILE、WSL env 无 `DEEPSEEK_API_KEY`；memory/image 输入件 0 |
| 07:54 | 科学动机钉死（07:54 steer） | §1.0 + frontmatter `motive_{cn,en}`：前置判断"只能偏好训练"不可靠（槽位放错）→ 已知偏好训练有效 / 未知能否完全替代 → 目标=证明可替换性 |
| 07:57 | B10 key 落盘 | Windows 用户环境存在 `DEEPSEEK_API_KEY`（仅记长度 35，未打印）；`E:\Envs\JitRL_LoRA_Env\.secrets\deepseek_key`（36 B，`/inheritance:r` → `ALWM18R2\Admin:(F)`）；WSL 用 `DEEPSEEK_API_KEY_FILE=/mnt/e/.../.secrets/deepseek_key` |
| 07:58 | selector 合成输入尝试（**作废**） | 依"不合成数据"原则取消；记录坑：`--candidates` 相对路径须以 prefmem-libero 根为 cwd |
| 07:59 | 判定（**已被 08:07 行取代**） | 当时 memory/image=0 → B10 暂开 |
| 08:02-08:03 | mission 包落盘复核 | `mission_info.yaml` **8/8**；missions 72 / 547,303,404 + assets 917 / 721,608,920 = **989 / 1,268,912,324**（与 Admin 逐项相等）；8 条 yaml MD5 已 pin；WSL 副本同数 |
| 08:03 | 执行器代码补拉（.109 只读） | `PolicyEval/ Preference/ Utility/ configs/ libero/{benchmark,utils,configs,lifelong}`；`base_object.py` / `hope_objects.py` / `objects/__init__.py` 与 .109 **MD5 完全一致** |
| 08:04 | 真 mission selector | `3-1_006_1` → `b9_decision_006.json`（`selected=pref`，4 步 plan） |
| 08:05 | 真 mission hop 尝试 #1 | `run_plan_with_vla.py` 起 env 失败：`KeyError: green_butter`（`import libero` 被 site-packages stock 遮蔽） |
| 08:06 | hop 尝试 #2 | 补 `openpi_client` 路径后前进到缺 `msgpack`（已装入 JitPi05 venv）；env-half 仍未过 |
| 08:07:40 | **B10 → RESOLVED** | 真截图 + 空 memory（cold start）跑通 selector：DeepSeek 返回真实场景 JSON，`SELECTOR_EXIT=0`，decision → `b10_decision_dryrun.json`；空 memory ⇒ Q≡0.5（plumbing-only 口径已声明） |
| 08:14 | 干净 venv 构建 | `~/projects/libero_rt/.venv`（**无 libero**）+ `robosuite==1.4.1`（含 `single_arm_env.py`）/mujoco/bddl/easydict/scipy/skimage/cloudpickle/gym==0.26.2/imageio/msgpack；`libero.__file__=None`（fork namespace ✓），但 `green_butter in OBJECTS_DICT` **False** |
| 08:15 | 参考运行时核查 | `/nvme-ssd/yanghaobo/pi05g_ws/libero_venv` 在 .109 **不存在**（无法取 freeze 作 pin） |
| 08:16 | B9 判定 | **部分清除**：missions/client-code/selector/server-half ✓；**env-half ✗** → STATUS 保持 BLOCKED |
| 08:22 | B9c 注册（数据） | `configs/task_suites.yaml#single-pref.missions` 追加 5 个缺失 id（40→45），现含 A 臂全 8 个；未改代码 |
| 08:23 | **env 根因定位+修复** | WSL `libero/` 原为 vanilla clone（registry 50 项）；E: fork 覆盖后 `objects/__init__.py 632ead5e…→00276ed3…`，`OBJECTS_DICT=991`、`green_butter=True`、`get_object_fn→GreenButter` |
| 08:24 | 真 mission 跑到 VLA 调用 | `run_plan_with_vla.py` 成功建 env 并进入 rollout；服务端报 `KeyError: 'observation/image'`（`libero_policy.py:52`） |
| 08:26 | 协议缝反证 | 同服务手工 4 键 / 5 键（含 `observation/eef_ori_mat`）均 `OK actions(10,7)`；客户端 openpi_client 与本 checkout 一致、msgpack 1.2.4 |
| 08:27 | B9 收口（第一版） | **部分清除**：missions/client/selector/299/env 建场景 ✓；协议缝待查 |
| 08:36 | shim 观测钩子上线 | `serve_pi05_local_config.py` 增加 `infer#N keys=...` 日志 + alias 归一化 |
| 08:37 | **协议缝根因** | 服务端第一条消息 = `{'__end__': True}`（客户端 `shutdown_server()` 哨兵），被误当 obs → `KeyError observation/image`；手工 4/5 键探针均 OK，故 schema 无问题 |
| 08:38 | shim 修哨兵 | `__end__` 短路返回，不再 infer |
| 08:39 | 直接跑 client（诊断） | `run_client.sh <runfile> --conditions pref`：`CLIENT_RC=0` 但 `policy_eval: done: 0 ok, 1 failed`；首个异常 = robosuite `binding_utils.py:521` `AssertionError`（reset 阶段） |
| 08:41 | pin 尝试失败 | `mujoco==2.3.7` 在 py3.12 上 `Failed to build … MUJOCO_PATH environment variable is not set`（无 cp312 wheel） |
| 08:42 | 根因结论 | 运行时栈不兼容：`mujoco 3.17.0 + robosuite 1.4.1`；.109 fork pycache 为 **cpython-310** → 需 py3.10 runtime venv |
| 08:43 | bddl 布局核查 | mission bddl 实际在 `bddl_files/libero_10/…`；本地 **361** 个 bddl = .109 361（一致） |
| 08:45-09:00 | **py3.11 运行时建成** | `libero_rt311`：`robosuite 1.4.1 + mujoco 2.3.7 + numpy<2`（`STACK 2.3.7/1.26.4/1.4.1`）；mission env `ENV_OK` |
| 09:00-09:05 | 代码树继续补齐 | `TaskGen`/`TaskEval`/`tools`/`templates`/`visualizer_v2`/`scripts`/`setup.py`/`requirements.txt` + `libero/libero/{benchmark,utils}`；`import TaskGen` → TASKGEN_OK |
| 09:07 | **当前阻塞** | `import libero.libero.utils` → 触发 `libero.libero.benchmark` 导入期 bddl 扫描 → `FileNotFoundError …libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_1.bddl`；该文件 **.109 也没有**（361=361）→ fork 不发 stock bddl 全集 |
| 09:14 | 关闭 eager bddl 扫描 | 最小 infra patch：`benchmark/__init__.py::grab_language_from_filename` 加 `try/except FileNotFoundError`（由文件名派生 language）；`import libero.libero.utils` → PKG_OK |
| 09:16-09:21 | **8 mission dry-run** | `run_client.sh` + 直连 runfile（8 ids, max_steps 600, replan 5, seed 7, 1 trial）：`done: **8 ok, 0 failed**`，`CLIENT_RC=0`；8 个 `rollout_summary.yaml` 落到 `artifacts/hf_pull_logs/b9_run8_summaries/` |
| 09:21 | SR（链路验证口径） | `3-2_032_7=1.0`、`3-2_032_8=1.0`，其余 6 个 `0.0`（单 trial，不作科学结论） |
| 09:22 | **B9 → RESOLVED；STATUS → READY** | open=[]；正式开跑仍受 §12.1 五项 Pre-launch 条件约束 |
| 09:53-10:09 | **并发写入告警（非本岗）** | `local_shims/` 出现他人/他岗文件：`run_chain_ab.sh`(09:53)、`compare_arms.py`(09:55)、`run_local_b_arm.py`(10:00)、`watch_v2_compare.sh`(10:06, md5 `642814b7…`)、`compare_arms_v2.py`(10:09+, md5 `57e296ed…` 早先快照)；根目录另有 `prereg_jitrl_vs_lora_acc[_v2].md`。**本轮未创建/未修改其中任何文件**，仅记录，不移动不删除 |
| 10:0x-10:2x | v2 selector 依赖修复 | 4B 需 `transformers==5.5.4`（4.53.2 不识别 `qwen3_5`）；CPU 栈补 `torch 2.15.0+cpu`、`torchvision 0.29.2+cpu`；卸载 peft（与 tf5 冲突），helpers 改为内联复用 |
| 10:26 | 单 mission 验证 | `libero_006_1`：`model_params=4539265536`、`logit_std 0.6875`、`pi0_std 0.298`、`entropy 0.503 nats`、`Â var 0`、selected=1；GPU 15465→15605 MiB（A 未受影响） |
| 10:3x | **8/8 mission selector dry-run** | 8 个决策 JSON：`artifacts/hf_pull_logs/b_sel4b_decision_<mission>.json`；logit_std = 0.125/0.5625/1.625/1.9375/2.6875/0.125/3.0/1.4375；`Â var` 全 0（冷启动合法）；`SEL8_RC=0`；GPU 15527→15539 MiB |
| 10:48 | 三 venv 基线冻结 | `libero_rt311` 95 包 `d9efcd2b…`；`openpi` 242 包 `55b1afcb…`；`JitPi05` 230 包 `88cb7c03…`（只读导出） |
| 10:49 | **隔离 selector venv 建成** | `~/projects/jitrl_sel4b/.venv`（py3.11）：torch 2.15.0+cpu / tv 0.29.2+cpu / tf 5.5.4 / cuda False；**39 包 / freeze `58d369f9…`** |
| 10:5x | **8/8 selector dry-run（新 venv）** | `SEL8_RC=0`；8 个决策 JSON `b_sel4b_newvenv_<mission>.json`；GPU 15621→15636 MiB（A 未受影响） |
| 10:57 | 三 venv 前后对比 | `libero_rt311`/`openpi`/`JitPi05` **全部 UNCHANGED**（同 sha256、同包数） |
| 11:0x | 规范 freeze 重算 | 两 venv 均**无 pip** → Lead 的 `python -m pip freeze --all` 不可执行（安装 pip 会改动冻结 venv）；改用 `uv pip freeze`（同数据源）+排序+LF+sha256：rt311 `d9efcd2b…`(95)、sel4b `58d369f9…`(39) |
| 11:05 | **pip 格式 canonical 实算** | 以 `uvx --from pip pip freeze --all --path <sp>`（只读）得：rt311 `45a0be68…`(95)、sel4b `be2d37bc…`(39)；venv 未被修改 |
| 11:06 | 变体复现测试 | pip 未排序 `b6cfca0e…`、pip CRLF `3cfd39c3…`、pip+`pip==` `84596971…`、uv 三变体 —— **均 ≠ `c0a21a38…`** → 该旧值 void（方法/时点差异） |
| 11:0x | 方法敏感性对照 | 同一包集：未排序 `132281c9…`、CRLF `64df3961…`、追加 `pip==…` `1318f248…` → 均 ≠ Exp `c0a21a38…`（方法差异，非包漂移） |
| 11:0x | accelerate 事件 | `libero_rt311` 内 `accelerate==1.15.0` dist-info mtime **10:01:12**（sel4b 10:49:05） |
| 11:0x | 资产/ckpt 指纹 | missions 72 / 547,303,404 B / md5-list `aff1bf8f…`；assets 917 / 721,608,920 B / `996ff351…`；ckpt 299 28 / 9,363,286,152 B / `977eb6e9…`（size-manifest `a9ee6806…`） |
