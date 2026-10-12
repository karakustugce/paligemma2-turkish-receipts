"""Dependency-free input validation, scoring, and paired statistics."""

import csv
import hashlib
import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


def parse_amount(value):
    """Only one explicit nonnegative amount; ambiguous output is a format error."""
    s = str(value).strip().replace("₺", "").strip()
    s = re.sub(r"(?i)\s*(?:TRY|TL)\s*$", "", s).strip()
    if not re.fullmatch(r"\d+(?:[.,]\d+)*", s):
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        group = "." if dec == "," else ","
        integer, fraction = s.rsplit(dec, 1)
        pattern = r"\d{1,3}(?:" + re.escape(group) + r"\d{3})+"
        if len(fraction) != 2 or not re.fullmatch(pattern, integer):
            return None
        s = integer.replace(group, "") + "." + fraction
    elif "," in s or "." in s:
        dec = "," if "," in s else "."
        if s.count(dec) != 1:
            return None
        integer, fraction = s.split(dec)
        if len(fraction) != 2:
            return None
        s = integer + "." + fraction
    try:
        return Decimal(s).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def load_manifest(path, image_root):
    """Reject missing labels, duplicate images, and paths outside the image directory."""
    root = Path(image_root).resolve()
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"doc_id", "image_path", "true_total"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV columns required: doc_id, image_path, true_total")
        source = list(reader)
    if not 30 <= len(source) <= 50:
        raise ValueError("The frozen evaluation set must contain 30–50 documents.")
    ids, hashes, docs = set(), set(), []
    for row in source:
        doc_id = row["doc_id"].strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", doc_id) or doc_id in ids:
            raise ValueError(f"Invalid or repeated doc_id: {doc_id}")
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"Image path leaves image_root: {doc_id}")
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError(f"Missing or unsupported image: {doc_id}")
        total = parse_amount(row["true_total"])
        if total is None:
            raise ValueError(f"Missing or ambiguous ground truth: {doc_id}")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Byte-identical duplicate image: {doc_id}")
        ids.add(doc_id); hashes.add(digest)
        docs.append({
            "doc_id": doc_id, "image": str(image), "sha256": digest,
            "true_total": str(total),
            "capture_quality": row.get("capture_quality", "unspecified") or "unspecified",
            "length_group": row.get("length_group", "unspecified") or "unspecified",
            "merchant_group": row.get("merchant_group", "unspecified") or "unspecified",
        })
    return sorted(docs, key=lambda x: x["doc_id"])


def score_answer(raw_answer, truth):
    predicted = parse_amount(raw_answer)
    if predicted is None:
        return {"predicted_total": None, "correct": False, "error_type": "format_or_unreadable", "absolute_error": None}
    correct = predicted == Decimal(truth)
    return {"predicted_total": str(predicted), "correct": correct,
            "error_type": "correct" if correct else "wrong_total",
            "absolute_error": float(abs(predicted - Decimal(truth)))}


def wilson(successes, n, z=1.96):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Invalid counts")
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0.0, center-radius), min(1.0, center+radius)


def paired_stats(rows, bootstrap_samples=10000, seed=2026):
    groups = {224: {}, 448: {}}
    for row in rows:
        resolution = row["resolution"]
        if resolution not in groups or type(row["correct"]) is not bool:
            raise ValueError("Invalid resolution or correctness flag")
        if row["doc_id"] in groups[resolution]:
            raise ValueError("Duplicate model/document pair")
        groups[resolution][row["doc_id"]] = row
    if not groups[224] or groups[224].keys() != groups[448].keys():
        raise ValueError("Incomplete paired run; no headline comparison generated")
    deltas = []
    only224 = only448 = both = neither = 0
    for doc_id in sorted(groups[224]):
        a, b = groups[224][doc_id], groups[448][doc_id]
        if (a["sha256"], a["true_total"]) != (b["sha256"], b["true_total"]):
            raise ValueError("Image or ground truth differs across models")
        x, y = a["correct"], b["correct"]
        only224 += int(x and not y); only448 += int(y and not x)
        both += int(x and y); neither += int(not x and not y)
        deltas.append(int(y)-int(x))
    n = len(deltas); discordant = only224 + only448
    # Exact two-sided McNemar / binomial test on discordant pairs.
    pvalue = 1.0 if not discordant else min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(only224, only448)+1)) / 2**discordant)
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(deltas, k=n))/n for _ in range(bootstrap_samples))
    return {"n": n, "both_correct": both, "both_wrong": neither,
            "only_224_correct": only224, "only_448_correct": only448,
            "accuracy_difference_448_minus_224": sum(deltas)/n,
            "paired_bootstrap95": [boot[int(.025*bootstrap_samples)], boot[min(int(.975*bootstrap_samples), bootstrap_samples-1)]],
            "mcnemar_exact_p": pvalue, "bootstrap_seed": seed, "bootstrap_samples": bootstrap_samples}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
