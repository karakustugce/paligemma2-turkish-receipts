"""Download the curated public-source sample; do not redistribute raw images."""

import argparse
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def prepare(output, draft_manifest):
    draft = Path(draft_manifest)
    with draft.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = Path(output); images = out / "images"; images.mkdir(parents=True, exist_ok=True)
    def download(row):
        path = (images / row["image_path"]).resolve()
        if not path.is_relative_to(images.resolve()):
            raise ValueError("Image path leaves the private data directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == row["source_sha256"]:
            return
        url = "https://www.kaggle.com/api/v1/datasets/download/muhammedesaddoan/turkish-receipts-data/" + urllib.parse.quote(row["source_file"], safe="")
        with urllib.request.urlopen(url, timeout=60) as response:
            if not response.headers.get("Content-Type", "").startswith("image/"):
                raise ValueError("Dataset endpoint did not return an image")
            content = response.read()
        if hashlib.sha256(content).hexdigest() != row["source_sha256"]:
            raise ValueError("Source image changed; do not silently mix dataset versions")
        path.write_bytes(content)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, rows))
    (out / "manifest_draft.csv").write_bytes(draft.read_bytes())
    print(len(rows), "real images downloaded. Human-check draft totals and save manifest.csv before inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    parser.add_argument("--draft-manifest", required=True)
    args = parser.parse_args()
    prepare(args.out, args.draft_manifest)
