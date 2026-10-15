# KuaiRand 同时间反馈记录报错：原因与修复

日期：2026-10-13。

## 原因

旧准备程序把 `(user_id, video_id, time_ms, tab)` 作为唯一事件键。用户在官方 Pure 日志第 130066 行遇到同键反馈不同的记录，触发了 `Duplicate event has inconsistent feedback`。

官方将 time_ms 定义为交互时间戳，并未将上述四元组声明为唯一事件 ID。因此同键不够证明两行是同一观察，也不够证明数据损坏。旧程序的去重假设过于严格。

## 修复规则

使用完整 CSV 观察内容的 SHA256 做重复判断；必需 ID/二值字段规范化后计算，其他字段保留。is_rand 不被忽略；随机与普通曝光来源不同的记录不会强制合并。

- 内容相同的观察去重。
- 同四元组但反馈、观看时间或其他字段不同的观察保留。
- 同一训练用户—物品只在所有保留观察都没有点击/正向响应时保留弱负证据；已点击但未达到所选正向规则的记录保持未知，不因较早的未点击记录被赋予弱负标签。
- 记录 coarse_key_collision_rows、coarse_feedback_disagreement_rows 和少量来源示例。
- 训练用户—物品仍按训练期聚合；任何已认可训练正交互都排除负采样资格。
- 同一行内部认可正信号与明确拒绝冲突，仍不自动给正负标签。
- 时间切分、未来反馈隔离、词表和独立评价规则保留。

这不是简单注释掉异常或任意保留第一条记录。新准备清单的 protocol.deduplication 为 `full_observation_v2`，产物指纹会改变。

## 服务器更新

本次只需替换准备脚本。先在服务器备份旧文件：

```bash
cd /mnt/nfs0/zhaob_node2/projects/neu_pro/ml_test_refactor
cp code/prepare_kuairand.py code/prepare_kuairand.py.before_dedup_fix
```

从本地上传本次修复后的 `code/prepare_kuairand.py` 到服务器同名位置。使用 SCP、SFTP 或平台文件上传功能；不要覆盖原始数据或已有运行产物。

也可上传 `dist/kuairand_duplicate_fix.tar.gz` 及其 SHA256 文件到服务器家目录，校验后在项目目录解压。该包只有准备脚本、回归测试和相关文档，保留数据、配置与运行目录：

```bash
cd ~
sha256sum -c kuairand_duplicate_fix.tar.gz.sha256
cd /mnt/nfs0/zhaob_node2/projects/neu_pro/ml_test_refactor
tar -xzf ~/kuairand_duplicate_fix.tar.gz -C .
```

新脚本应包含 `event_signature` 与 `store_observation`：

```bash
grep -n 'def event_signature\|def store_observation' code/prepare_kuairand.py
```

## 重新准备并开始实验

使用新输出目录，保留旧准备失败时的诊断目录：

```bash
python code/prepare_kuairand.py \
  --raw-root data/raw/KuaiRand-Pure/data \
  --output data/kuairand_stage1_fixed \
  --train-end 20220428 \
  --val-end 20220503 \
  --positive-rule engagement \
  --eval-source random \
  --min-train-positives 2

python -m json.tool data/kuairand_stage1_fixed/manifest.json
```

确认出现 PREPARED、audit 中划分非空，随后先试跑：

```bash
python scripts/run_stage1.py \
  --dataset-dir data/kuairand_stage1_fixed \
  --output-root runs/stage1_smoke_fixed \
  --epochs 2 --seeds 2020 --device cuda
```

试跑成功后使用新的正式套件目录：

```bash
python scripts/run_stage1.py \
  --dataset-dir data/kuairand_stage1_fixed \
  --output-root runs/stage1_v1_fixed \
  --config configs/stage1.json --device cuda
```

已存在对应新目录时继续选择不同名称。不要混用旧数据/旧源码的检查点；恢复与最终评价会检查指纹。本次原始训练尚未开始，因此不需要恢复旧权重。

来源：[KuaiRand 官方字段说明](https://kuairand.com/)。

已在官方 MD5 一致的原始文件复现：第 130065 和 130066 行四元组相同，但 is_click 分别为 0/1，play_time_ms 分别为 1934/7250。这能确认旧唯一键假设不成立；不据此猜测两行属于重复曝光还是反馈更新。

完整数据准备已通过：读取 2,622,668 行，去除 21,997 条相同观察，保留 1,486 条同四元组不同观察，其中 114 条存在监督标记差异。按默认协议得到 451,741 条训练正交互、30,593 条验证目标、45,169 条测试目标，三个划分的交互交集为零。新增回归测试后的全套 20 项测试通过。

在该完整准备数据上另做 CUDA 功能检查：可靠采样、512 个训练样本、一个 epoch、完整 validation 通过，loss 和评价均为有限值。该检查不代表正式预算训练或推荐性能提升。
