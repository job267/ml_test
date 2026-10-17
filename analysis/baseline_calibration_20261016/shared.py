"""Utilities confined to the baseline study; original training code stays frozen."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def gpu_guard():
    if Path(sys.prefix).name != "gcn":
        raise RuntimeError("Use the existing gcn environment")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Require PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU required")
    physical = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    visible = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if physical.lower() != visible.lower():
        raise RuntimeError("CUDA device is not physical GPU 0")
    return {"gpu0_uuid": physical, "gpu": torch.cuda.get_device_name(0),
            "python": sys.executable, "torch": str(torch.__version__),
            "cuda_visible_devices": "0", "cuda_device_order": "PCI_BUS_ID"}


def history(path):
    if not Path(path).exists():
        return []
    result = []
    for line in Path(path).read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            break  # Writer may be appending the last line.
    return result
