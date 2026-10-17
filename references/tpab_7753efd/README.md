# 固定版本的公开参考文件

来源：<https://github.com/hsyoo32/tpab_>，固定 commit 为 `7753efd0a8cf3aaed9b150a3d835dbaa8fe0e50d`。本目录仅保存来源、字节数、Git blob SHA1 与 SHA256 清单；作者代码、交互数据和图缓存由原仓库提供。

基线校准和实现核对使用 `external/tpab_7753efd/`，该目录被 Git 忽略。已经运行实验的服务器继续使用现有文件。下面仅用于新的克隆目录，不在运行中的实验目录执行 checkout。

从项目根目录执行，继续使用已有 `gcn` 环境：

```bash
git clone --no-checkout https://github.com/hsyoo32/tpab_.git external/tpab_7753efd
git -C external/tpab_7753efd checkout --detach 7753efd0a8cf3aaed9b150a3d835dbaa8fe0e50d
```

核验清单中的全部 22 个文件，并将完全一致的清单放到驱动要求的位置：

```bash
conda run --no-capture-output -n gcn python - <<'PY'
from pathlib import Path
import hashlib
import json

root = Path.cwd()
manifest = root / "references/tpab_7753efd/provenance.json"
vendor = root / "external/tpab_7753efd"
payload = manifest.read_bytes()
records = json.loads(payload)
assert records["commit"] == "7753efd0a8cf3aaed9b150a3d835dbaa8fe0e50d"
for record in records["files"]:
    data = (vendor / record["path"]).read_bytes()
    header = b"blob " + str(len(data)).encode() + b"\0"
    assert len(data) == record["bytes"], record["path"]
    assert hashlib.sha1(header + data).hexdigest() == record["git_blob_sha1"], record["path"]
    assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
target = vendor / "provenance.json"
if target.exists():
    assert target.read_bytes() == payload, "Existing provenance differs; inspect it before continuing"
else:
    target.write_bytes(payload)
print(f"Verified {len(records['files'])} pinned reference files")
PY
```

本项目的原始 KuaiRand-Pure 日志、prepared 数据与运行检查点另行准备，见[数据准备协议](../../docs/stage1_implementation.md)和[服务器工作流程](../../docs/github_server_workflow.md)。GitHub 中的状态 JSON 是带时间戳的元数据快照，不能代替数据或检查点，也不能仅凭快照在另一台机器恢复正在运行的队列。
