# 使用 GitHub 管理服务器代码

仓库：<https://github.com/job267/ml_test>。默认分支：`main`。

代码、配置、测试、研究文档和第一轮验证分析由 Git 管理。`data/`、`runs/`、`checkpoints/`、`dist/`、压缩包、模型权重和环境文件在本地或服务器独立保存。GitHub 不代替实验数据与检查点备份。

## 1. 为现有服务器建立 Git 工作目录

服务器已有的实验目录是：

```text
/mnt/nfs0/zhaob_node2/projects/neu_pro/ml_test_refactor
```

在它旁边克隆一个新的目录，保留现有数据和实验产物。以下命令在服务器 Bash 中逐行执行；继续使用已有的 `gcn` 环境。

```bash
conda activate gcn
```

```bash
cd /mnt/nfs0/zhaob_node2/projects/neu_pro
```

```bash
git clone https://github.com/job267/ml_test.git ml_test_refactor_git
```

```bash
cd /mnt/nfs0/zhaob_node2/projects/neu_pro/ml_test_refactor_git
```

如果服务器已配置 GitHub SSH，也可在首次克隆时使用 `git@github.com:job267/ml_test.git`。私有仓库需要当前登录身份具有读取权限。

若新目录名已存在，先检查其中是否已有代码，或选择另一个新目录名；不要覆盖旧实验目录。

复用原目录的数据与运行结果，无需复制数据：

```bash
ln -s ../ml_test_refactor/data data
```

```bash
ln -s ../ml_test_refactor/runs runs
```

仅在克隆目录中不存在对应的 `data`、`runs` 路径时执行这两条命令。符号链接的目标仍位于旧实验目录，后续继续保留该目录。

确认环境、代码版本和结果可访问：

```bash
git status --short
```

```bash
git log -1 --oneline
```

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

```bash
python -m json.tool runs/stage1_v1/suite.json
```

首次由 GitHub 克隆后可运行现有检查：

```bash
python -m pytest -q
```

## 2. 接着完成第一轮的验证参照

进入新克隆目录后，直接运行已经随仓库上传的工具，不再传输工具压缩包：

```bash
python analysis/stage1_validation_20261015/evaluate_validation_references.py --suite-dir runs/stage1_v1 --output runs/stage1_v1/validation_references.json --device cpu
```

工具会校验原套件数据和源码版本，只计算 validation 上的随机期望与训练热门推荐结果。若已有同名输出，保留它，使用另一个新的输出名称。

套件保存的 dataset_dir 是原目录下的绝对路径；保留原数据目录即可继续使用。这也是第一次迁移选择相邻克隆和符号链接、而不移动旧目录的原因。

## 3. 后续更新代码

当前目录不再进行训练、恢复或模型评估时，再更新工作区：

```bash
cd /mnt/nfs0/zhaob_node2/projects/neu_pro/ml_test_refactor_git
```

```bash
git status --short
```

```bash
git pull --ff-only origin main
```

```bash
git log -1 --oneline
```

如果 status 显示服务器本地改动，先保存并明确如何处理这些改动，再拉取；不使用强制重置覆盖它们。如果 `pull --ff-only` 提示分支分叉，先检查本地提交，不强制推送。

每个新实验使用新的输出目录，记录对应的 commit 和环境；原有训练入口也会自动记录 commit、源码 SHA256、数据 SHA256 和配置。更新依赖时先核对 requirements 和实验协议，不能把两种环境的结果直接混为同一对照。

## 4. 旧检查点的源码版本校验

现有第一轮实验对 `code/*.py` 和 `scripts/*.py` 记录逐文件 SHA256，其中一部分研究脚本使用 CRLF 换行。`.gitattributes` 保留这些文件的原始字节，使 Linux 克隆得到的内容与原上传包一致。

首次上传已检查 Git 暂存内容与第一轮的 16 个源码指纹一致。后续改动任意被记录的源码或新增同目录 Python 文件，都可能触发恢复/最终评价的版本检查。研究改进应使用新的实验目录，并保留旧版本以便评价旧模型。

不要通过移除源码校验、修改旧 suite 指纹或删除测试计划来兼容新代码。需要评价旧模型时，在另一个目录检出产生它的原始版本，并使用原 prepared 数据。

## 5. 服务器上的配置实验

正式配置提交到 Git；临时参数可放在被忽略的运行目录，例如 `runs/local_configs/`。从基线配置复制后，只修改此次研究的参数，保留独立记录。

```bash
mkdir -p runs/local_configs
```

```bash
cp configs/stage1.json runs/local_configs/stage1_beta_pilot.json
```

新实验的 `--config` 指向新配置，`--output-root` 指向新的运行目录。已有 `runs/stage1_v1` 继续作为第一轮原始结果保存。
