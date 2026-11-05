"""Upload the demo to a Hugging Face Space.

Stages exactly the files the Space needs (see "Deploy to a Hugging Face Space" in
`demo/README.md`) and uploads them to the Space. The Space installs this package from
GitHub; the staged `requirements.txt` pins it to `--ref` (a release tag or a commit)
instead of `main`, so the Space keeps working when the repository changes. Nothing
from `data/` or `outputs/` is ever staged.

Usage:
    HF_TOKEN=... python scripts/sync_space.py --space m-555/explain-to-learn --ref v1.0.0
    python scripts/sync_space.py --stage-only build/space --ref v1.0.0   # to inspect

The GitHub workflow `.github/workflows/space-sync.yml` runs it with the `HF_TOKEN`
secret and the `HF_SPACE_ID` variable (docs/release.md, step 5).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parents[1]

# Source in this repository -> path in the Space.
FILES = {
    "demo/app.py": "app.py",
    "demo/stubs.py": "stubs.py",
    "demo/requirements.txt": "requirements.txt",
    "demo/space_card.md": "README.md",
    "configs/systems.yaml": "configs/systems.yaml",
    "grammar_graph/concepts.yaml": "grammar_graph/concepts.yaml",
    "grammar_graph/errant_map.yaml": "grammar_graph/errant_map.yaml",
    "grammar_graph/README.md": "grammar_graph/README.md",
}
PACKAGE_SOURCE = "git+https://github.com/m-555/explain-to-learn@main"


def pin_package(requirements: str, ref: str) -> str:
    """Point the package requirement at `ref` instead of `main`."""
    if PACKAGE_SOURCE not in requirements:
        raise ValueError(f"demo/requirements.txt no longer contains {PACKAGE_SOURCE}")
    return requirements.replace(PACKAGE_SOURCE, PACKAGE_SOURCE.removesuffix("main") + ref)


def stage(dest: Path, ref: str | None, root: Path = ROOT) -> list[str]:
    """Copy the Space's files into `dest` and return their paths in the Space."""
    if dest.exists():
        shutil.rmtree(dest)
    for source, target in FILES.items():
        path = dest / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / source, path)
    if ref:
        requirements = dest / "requirements.txt"
        text = requirements.read_text(encoding="utf-8")
        requirements.write_text(pin_package(text, ref), encoding="utf-8")
    return sorted(FILES.values())


def upload(folder: Path, space: str, token: str, message: str) -> None:
    """Upload the staged folder to the Space (other files in the Space are kept)."""
    from huggingface_hub import HfApi

    HfApi(token=token).upload_folder(
        folder_path=str(folder), repo_id=space, repo_type="space", commit_message=message
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload the demo to a Hugging Face Space.")
    parser.add_argument("--space", default=os.environ.get("HF_SPACE_ID"), help="user/name")
    parser.add_argument("--ref", help="release tag or commit to pin the package to")
    parser.add_argument("--stage-only", type=Path, help="stage into this folder, no upload")
    args = parser.parse_args(argv)

    if args.stage_only:
        files = stage(args.stage_only, args.ref)
        print(f"Staged {len(files)} files in {args.stage_only}")
        return 0
    if not args.space:
        parser.error("give --space or set HF_SPACE_ID")
    if not args.ref:
        parser.error("give --ref (a release tag or commit), so the Space is pinned")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; create a token with write access to the Space")
    staging = ROOT / "build" / "space"
    files = stage(staging, args.ref)
    upload(staging, args.space, token, f"Sync demo from {args.ref}")
    print(f"Uploaded {len(files)} files to https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
