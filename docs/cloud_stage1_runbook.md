# 云服务器操作指南：上传 → 环境 → 数据 → 第一阶段实验

日期：2026-10-13。以下步骤对应本项目已提供的独立第一阶段实验入口。云平台不限；推荐先选择带 NVIDIA GPU 的 Ubuntu 22.04、Python 3.10 环境。

实验范围是离线可靠负采样和课程调度，不包含在线真人展示或第二阶段残差结构。完整协议先阅读 [第一阶段实现说明](stage1_implementation.md)。

## 1. 本地准备上传包

在 Windows PowerShell 中进入当前项目：

```powershell
Set-Location 'D:\code\MLearn\LightGCN_PyTorch2_compat_refactor\ml_test_refactor'
python -m pytest -q
python scripts/package_project.py --output dist/lightgcn_stage1.tar.gz
```

压缩包只含源码、测试、配置和文档。不会上传原始数据、历史权重、虚拟环境或 Git 目录。旁边的 `.sha256` 文件可核对传输完整性。

如同名包已存在，打包程序拒绝覆盖；使用新的名称，例如 `dist/lightgcn_stage1_v2.tar.gz`，后续命令相应替换。

## 2. 获取服务器登录信息并上传

从租用平台获取公网地址、SSH 端口、用户名和登录方式；服务器应已安装 NVIDIA 驱动。下面三项替换为平台实际值：

```powershell
$serverAddress = 'YOUR_SERVER_ADDRESS'
$serverPort = 22
$serverUser = 'YOUR_SSH_USER'
scp -P $serverPort .\dist\lightgcn_stage1.tar.gz .\dist\lightgcn_stage1.tar.gz.sha256 "${serverUser}@${serverAddress}:~/"
ssh -p $serverPort "${serverUser}@${serverAddress}"
```

若平台要求密钥，为 scp/ssh 加 `-i 'C:\path\your_key'`。以下所有 Bash 命令在服务器终端执行。

## 3. 校验、解压与确认 GPU

```bash
cd ~
sha256sum -c lightgcn_stage1.tar.gz.sha256
mkdir -p ~/projects
tar -xzf lightgcn_stage1.tar.gz -C ~/projects
cd ~/projects/ml_test_refactor
nvidia-smi
python3 --version
```

首次使用上述目录；若服务器已有同名工作目录，应选择新的目标目录再解压，避免覆盖旧实验。数据和运行记录放在持久磁盘；确认平台是否在关机或释放实例时删除该磁盘。

`nvidia-smi` 显示的 CUDA 字样是驱动支持上限，不是 PyTorch 使用的运行时版本。若 GPU 不可见，先通过平台提供的驱动/镜像解决，再进行安装。

## 4. 建立隔离环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -m pip freeze > environment_cloud.txt
```

上述 CUDA 12.6 构建来自 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。需兼容驱动。官方也提供 2.6.0 的 cu124 等构建；若更换构建，记录实际环境，并对全部对照统一使用。

如缺少 venv、curl、tmux、git，可用平台提供的安装功能；具有 sudo 权限的 Ubuntu 22.04 用户可执行：

```bash
sudo apt-get update
sudo apt-get install -y python3-venv curl tmux git
```

先检查 Python 版本，本文以 3.10 为目标环境。不要将服务器其他项目的环境混入本实验。

## 5. 下载反馈数据

本轮使用 KuaiRand-Pure 做离线机制实验。其候选物品范围经过筛选；官方建议严格长序列/RL/OPE研究使用 1K 或完整版本，故当前 Pure 结果不用于宣称完整线上 RL 效果。

```bash
mkdir -p data/downloads data/raw
curl -L --fail --retry 3 https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz -o data/downloads/KuaiRand-Pure.tar.gz
echo '0820331067a3784d9691136f772b35a7  data/downloads/KuaiRand-Pure.tar.gz' | md5sum -c -
tar -xzf data/downloads/KuaiRand-Pure.tar.gz -C data/raw
find data/raw -name 'log_*.csv'
```

下载 URL 与 MD5 来自 [KuaiRand 官方页面](https://kuairand.com/)。若下载失败，可按官方页面下载后手工上传；不要跳过校验。若官方未来更新文件导致校验不同，核对新版发布信息并另存版本。

确认找到 Pure 的三类标准/随机日志，并用只包含这个版本的目录作为 raw-root。首次下载只有 Pure 时，下列 `data/raw` 搜索也可使用。

## 6. 固定数据协议并准备

```bash
python code/prepare_kuairand.py --raw-root data/raw --output data/kuairand_stage1 --train-end 20220428 --val-end 20220503 --positive-rule engagement --eval-source random --min-train-positives 2
python -m json.tool data/kuairand_stage1/manifest.json
```

检查输出 audit：训练/验证/测试是否非空，明确负证据有多少，弱证据有多少，未知用户/物品过滤数量、重叠与零度物品。准备目录存在时命令报错，这是避免静默改写数据的保护。

若任何划分为空，先审计原始日期与字段，再改变协议并输出到新目录；全部方法必须使用同一份准备结果。未来评价标签不得预先进入训练图或负证据。

原始/准备数据可另行备份；manifest 保存原 CSV 和产物 SHA256。实验启动时逐文件校验。

## 7. 检查测试与短程试跑

```bash
python -m pytest -q
python scripts/run_stage1.py --dataset-dir data/kuairand_stage1 --output-root runs/stage1_smoke --epochs 2 --seeds 2020 --device cuda
```

这会串行跑四组，每组两轮，默认训练总样本量保持正常。短程套件只验证训练与 validation，不做 test。

如需要更小的功能试跑，可单独指定样本数：

```bash
python code/experiment.py --dataset-dir data/kuairand_stage1 --run-id reliable_tiny --sampler reliable --epochs 2 --samples-per-epoch 4096 --device cuda
```

`samples-per-epoch` 改变训练预算，因此 tiny 结果只作功能检查。正式四组使用配置中的 0，即唯一训练交互数。不要把 tiny 的结果与正式基线比较。

检查每组 `.log` 和 `metrics.jsonl` 中有限 loss、validation、采样时间、候选打分数和峰值显存。根据短程耗时估算预算，当前尚无真实数据的训练时长保证。

## 8. 正式运行四组、三个种子

使用新的输出目录，默认配置 `configs/stage1.json` 含四组、2020/2021/2022 三种子、1000 Epoch，固定预算且无早停。该配置是初始对照，不是已经确定的最优参数。

```bash
tmux new -s stage1
cd ~/projects/ml_test_refactor
source .venv/bin/activate
python scripts/run_stage1.py --dataset-dir data/kuairand_stage1 --output-root runs/stage1_v1 --config configs/stage1.json --device cuda
```

每次仅运行一组，避免组间 GPU 争抢。控制终端会显示 START/DONE；详细训练输出写入对应日志。

按 `Ctrl+B` 再按 `D` 脱离 tmux，SSH 断开后任务继续。重新登录：

```bash
tmux attach -t stage1
```

另一个终端查看状态：

```bash
cd ~/projects/ml_test_refactor
tail -n 3 runs/stage1_v1/uniform-seed2020.log
nvidia-smi
```

研究超参数时复制 JSON 到新文件并使用新输出目录，只看 validation，保持所有对照一致。不要反复用 test 选择候选数、证据权重或课程长度。

## 9. 中断后的恢复

套件有检查点后，可使用与原启动相同的数据、配置、epochs 等参数，增加 `--resume`：

```bash
python scripts/run_stage1.py --dataset-dir data/kuairand_stage1 --output-root runs/stage1_v1 --config configs/stage1.json --device cuda --resume
```

已完成组跳过，未完成组从 `last.pt` 恢复模型、优化器和随机数状态。源代码或配置改变时拒绝将不同实验混在一起。

若在第一轮检查点生成前失败，保留日志和目录，使用新的套件目录重新开始；不要伪造一个完成标记。GPU 设备/软件环境变化时，恢复不保证与原连续训练位级一致。

单运行恢复示例，需保留原训练参数：

```bash
python code/experiment.py --dataset-dir data/kuairand_stage1 --sampler reliable --epochs 1000 --device cuda --resume runs/research/reliable_tiny/last.pt --samples-per-epoch 4096
```

该例只说明恢复 tiny 的方式，依然不是正式预算实验。

## 10. 最终测试与汇总

全部正式组训练完成、验证方案选择结束后，再执行：

```bash
python scripts/evaluate_stage1.py --suite-dir runs/stage1_v1 --checkpoint last --device cuda
python scripts/summarize_stage1.py --suite-dir runs/stage1_v1
```

`last` 代表固定 1000 Epoch 的最终模型。`best_val` 是另一种预先统一约定的选择，不能查看 test 后再切换。首次 test 前写入 `final_test_plan.json`，冻结全组检查点及 SHA。

若最终测试中断，恢复同一方案：

```bash
python scripts/evaluate_stage1.py --suite-dir runs/stage1_v1 --checkpoint last --device cuda --resume
python scripts/summarize_stage1.py --suite-dir runs/stage1_v1
```

结果包括 `summary.csv`、`summary.json`、分组指标、均值/样本标准差和种子配对差异。单种子标准差标为 null；汇总不自动作统计显著性结论。

## 11. 保存并下载产物

服务器上打包运行结果与环境记录；数据原文件可独立备份：

```bash
tar -czf stage1_results.tar.gz runs/stage1_v1 environment_cloud.txt data/kuairand_stage1/manifest.json
sha256sum stage1_results.tar.gz > stage1_results.tar.gz.sha256
```

在本地另开 PowerShell，重新设置第 2 步的登录变量后下载：

```powershell
scp -P $serverPort "${serverUser}@${serverAddress}:~/projects/ml_test_refactor/stage1_results.tar.gz" .\
scp -P $serverPort "${serverUser}@${serverAddress}:~/projects/ml_test_refactor/stage1_results.tar.gz.sha256" .\
```

确认本地已收到并能读取检查点、日志、汇总及数据指纹后，再按云平台规则停止或释放实例。保存同版源码包与完整准备数据，才能之后独立复核。

## 12. 常见失败定位

| 情况 | 处理 |
|---|---|
| CUDA 不可用 | 查看 nvidia-smi、驱动和所装 torch 构建；正式命令使用 --device cuda 会明确报错 |
| 缺少 CSV 字段 | 确认是官方日志，未混入特征表或其他版本；加载器严格检查字段 |
| validation/test 为空 | 审计日期、正向规则及随机曝光记录；准备新协议目录后重跑全部组 |
| 数据 checksum 不符 | 使用原始准备产物；不要手改 TXT/CSV 后继续旧实验 |
| 输出目录已存在 | 使用新目录，或对匹配套件使用 --resume |
| 显存不足 | 降低 test_batch、score_batch 可控制评价/采样内存；若改变套件参数，统一新配置与目录 |
| 训练慢 | 先看 sampler_seconds、train_seconds；当前保留每 batch 全图传播，不能随意缓存训练图表示来加速 |
| source/config changed | 保留旧产物，建立新实验；检查点恢复主动拒绝混用 |
| Duplicate event has inconsistent feedback | 使用修复后的准备脚本；按[去重修复说明](kuairand_duplicate_fix.md)保留同时间不同观察，并生成新数据目录 |

本地测试通过只证明协议与代码行为，完整数据/GPU 结果仍需按以上步骤在服务器验证。
