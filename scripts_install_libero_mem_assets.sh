#!/bin/bash
# 把 LIBERO-Mem 的 bddl 与 init 文件铺进 libero 包目录。
#
# 前置：先把仓库浅克隆到 $SRC：
#   git clone --depth 1 -b main https://github.com/libero-mem/libero-mem.git ~/LIBERO-Mem
#
# 为什么要「互相复用」init 文件：原仓库只预生成了 4 个任务的 pruned_init。
# 已核对 BDDL 的 :init 谓词——任务 1–6 **完全同构**、任务 9 与 10 同构，
# 所以采样出来的初始状态互相合法。这顺带给出一个好性质：
# **任务 1/2/3/4/5/6 共享同一批初始状态**，是「同一初始状态、不同目标」的直接配对。
#
# 用法：bash install_libero_mem_assets.sh [LIBERO-Mem 仓库路径]
set -euo pipefail

SRC="${1:-$HOME/LIBERO-Mem}"
DST="${JITPI05_ROOT:-$HOME/projects/JitPi05}/third_party/libero-pro/libero/libero"

[ -d "$SRC/libero/libero/bddl_files/libero_mem" ] || {
  echo "找不到 $SRC/libero/libero/bddl_files/libero_mem" >&2; exit 1; }

mkdir -p "$DST/bddl_files/libero_mem" "$DST/init_files/libero_mem"
cp "$SRC"/libero/libero/bddl_files/libero_mem/*.bddl "$DST/bddl_files/libero_mem/"
cp "$SRC"/scripts/__init_data/*.pruned_init "$DST/init_files/libero_mem/"

B="$DST/bddl_files/libero_mem"
I="$DST/init_files/libero_mem"

# 1–6 场景同构 ⇒ 复用场景 1 的 init；9/10 同构 ⇒ 复用场景 9 的
for n in 2 3 4 5 6; do
  s=$(ls "$I" | grep "^KITCHEN_SCENE1_1_")
  t=$(ls "$B" | grep "^KITCHEN_SCENE1_${n}_" | sed 's/\.bddl$/.pruned_init/')
  cp "$I/$s" "$I/$t"
done
s9=$(ls "$I" | grep "^KITCHEN_SCENE1_9_")
t10=$(ls "$B" | grep "^KITCHEN_SCENE1_10_" | sed 's/\.bddl$/.pruned_init/')
cp "$I/$s9" "$I/$t10"

echo "bddl: $(ls "$B" | wc -l) 个，init: $(ls "$I" | wc -l) 个"
miss=0
for f in "$B"/*.bddl; do
  b=$(basename "$f" .bddl)
  [ -f "$I/$b.pruned_init" ] || { echo "  缺 init: $b"; miss=$((miss+1)); }
done
[ "$miss" -eq 0 ] && echo "每个 bddl 都有对应的 init ✓"
