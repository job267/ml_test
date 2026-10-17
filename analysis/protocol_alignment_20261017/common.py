"""Independent study utilities; frozen core and author files are read-only inputs."""
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VENDOR = ROOT / "external/tpab_7753efd"
OUTPUT = ROOT / "runs/protocol_alignment_20261017"
CONFIG = ROOT / "configs/protocol_alignment_20261017.json"
sys.path.insert(0, str(ROOT / "code"))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def history(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


def gpu_guard():
    import torch
    if Path(sys.prefix).name != "gcn" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Require existing gcn, PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU is required")
    physical = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    visible = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if physical.lower() != visible.lower():
        raise RuntimeError("Visible CUDA device is not physical GPU 0")
    return {"python": sys.executable, "torch": str(torch.__version__), "gpu0_uuid": physical,
            "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": "0", "cuda_device_order": "PCI_BUS_ID"}


def verify_author(names):
    provenance = read(VENDOR / "provenance.json")
    expected = {r["path"]: r["sha256"] for r in provenance["files"]}
    for name in names:
        if sha(VENDOR / name) != expected[name]:
            raise ValueError("Pinned author file changed: " + name)
    return {name: expected[name] for name in names}


def author_definitions(config, device):
    """Compile exact pinned class/functions, excluding import-time CLI/IO side effects.

    No function body is edited. Stub popularity is unused by vanilla LightGCN;
    held-out test labels and the author's future-popularity tables are never loaded.
    """
    import numpy as np
    import torch
    from torch import nn, optim
    from dataloader import BasicDataset
    verified = verify_author(["model.py", "dataloader.py", "Procedure.py"])
    world = SimpleNamespace(config=config, device=torch.device(device), topks=[10, 20, 50, 100])
    def minibatch(users, batch_size):
        for start in range(0, len(users), batch_size):
            yield users[start:start + batch_size]
    namespace = {"__name__": "pinned_author_alignment", "torch": torch, "nn": nn, "optim": optim,
                 "np": np, "math": math, "logging": logging, "world": world,
                 "BasicDataset": BasicDataset, "dataloader": SimpleNamespace(BasicDataset=BasicDataset),
                 "utils": SimpleNamespace(minibatch=minibatch, BasicDataset=BasicDataset)}
    model_tree = ast.parse((VENDOR / "model.py").read_text())
    nodes = [n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name in ("BasicModel", "LightGCN")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(VENDOR / "model.py"), "exec"), namespace)
    namespace["model"] = SimpleNamespace(LightGCN=namespace["LightGCN"])
    data_tree = ast.parse((VENDOR / "dataloader.py").read_text())
    cls = next(n for n in data_tree.body if isinstance(n, ast.ClassDef) and n.name == "DisenData")
    sampler = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_pair_bpr")
    exec(compile(ast.Module(body=[sampler], type_ignores=[]), str(VENDOR / "dataloader.py"), "exec"), namespace)
    proc_tree = ast.parse((VENDOR / "Procedure.py").read_text())
    functions = [n for n in proc_tree.body if isinstance(n, ast.FunctionDef) and n.name in ("Test", "computeTopNAccuracy")]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(VENDOR / "Procedure.py"), "exec"), namespace)
    return SimpleNamespace(model=namespace["LightGCN"], sample=namespace["get_pair_bpr"],
        evaluate=namespace["Test"], metrics=namespace["computeTopNAccuracy"], world=world, verified=verified)


def checkpoint(path, value):
    import torch
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)
