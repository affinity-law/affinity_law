"""Verify local weights, or explicitly download the exact recorded revisions.

Run from any directory. Network access occurs only with --download; existing
files are verified and never overwritten. Model code is not loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODELS = ("timesfm3_0", "moirai2_0")


def verify(path, provenance):
    if path.stat().st_size != provenance["weight_size"]:
        raise ValueError(f"Weight size differs from provenance: {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != provenance["weight_sha256"]:
        raise ValueError(f"Weight SHA256 differs from provenance: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Verify only (the default)")
    mode.add_argument("--download", action="store_true", help="Download missing weights from Hugging Face")
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    args = parser.parse_args()
    models = MODELS if args.model == "all" else (args.model,)
    for model in models:
        checkpoint = ROOT / model / "checkpoint"
        provenance = json.loads((checkpoint / "provenance.json").read_text(encoding="utf-8"))
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Restore the tracked model config: {checkpoint / 'config.json'}")
        target = checkpoint / "model.safetensors"
        if target.is_file():
            verify(target, provenance)
            print(f"PASS {model}: existing weight matches recorded SHA256")
            continue
        if not args.download:
            raise FileNotFoundError(f"Missing {target}; use --download to fetch the recorded revision")
        url = (f"https://huggingface.co/{provenance['repository']}/resolve/"
               f"{provenance['revision']}/model.safetensors")
        temporary = target.with_suffix(".safetensors.download")
        print(f"Downloading {model}: {provenance['repository']} @ {provenance['revision']}", flush=True)
        # Exclusive creation preserves any previous incomplete transfer for inspection.
        with temporary.open("xb") as destination:
            request = urllib.request.Request(url, headers={"User-Agent": "foundation-reproduction/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    destination.write(block)
        verify(temporary, provenance)
        # A hard link refuses to replace a file another process created meanwhile.
        os.link(temporary, target)
        temporary.unlink()
        print(f"PASS {model}: downloaded weight matches recorded SHA256")


if __name__ == "__main__":
    main()
