"""Package source, tests, configs and documentation; never include datasets or weights."""

import argparse
from pathlib import Path
import tarfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from research_common import sha256


def package(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = list(ROOT.glob("*.md")) + [ROOT / "requirements.txt", ROOT / ".gitignore"]
    for folder, suffixes in [
        ("code", {".py"}),
        ("scripts", {".py"}),
        ("tests", {".py"}),
        ("docs", {".md"}),
        ("configs", {".json"}),
    ]:
        files.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts
        )
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(files):
            if path.is_symlink():
                raise ValueError(f"Unexpected symbolic link: {path}")
            archive.add(
                path,
                arcname=str(Path("ml_test_refactor") / path.relative_to(ROOT)),
                recursive=False,
            )
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        digest + "  " + output.name + "\n", encoding="ascii"
    )
    print(f"PACKAGE {output.resolve()}\nSHA256 {digest}\nFILES {len(files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "lightgcn_stage1.tar.gz")
    package(parser.parse_args().output)
