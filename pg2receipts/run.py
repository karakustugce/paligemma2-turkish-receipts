"""GPU inference with pinned model revisions and resumable, validated output."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from . import MODELS, PROMPTS, __version__
from .core import load_manifest, read_jsonl, score_answer


def run(manifest, image_root, output, resume=False):
    docs = load_manifest(manifest, image_root)
    # Heavy libraries load only after data validation.
    import torch
    from huggingface_hub import HfApi
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Select a Colab T4 GPU runtime.")
    torch.manual_seed(2026)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    frozen = [{k: v for k, v in doc.items() if k != "image"} for doc in docs]
    canonical = json.dumps(frozen, ensure_ascii=False, sort_keys=True).encode()
    fingerprint = hashlib.sha256(canonical).hexdigest()
    source = b"".join(p.name.encode()+p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
    identity = {"manifest_fingerprint": fingerprint, "documents": frozen, "prompts": PROMPTS,
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "package_version": __version__, "dtype": "float16", "max_new_tokens": 32,
                "sampling": False, "seed": 2026, "attention": "eager"}
    cfg_path = out / "run_config.json"; rows_path = out / "predictions.jsonl"
    if cfg_path.exists():
        if not resume:
            raise ValueError("Output exists. Use --resume with the same frozen dataset or choose a new output directory.")
        cfg = json.loads(cfg_path.read_text())
        if any(cfg.get(k) != v for k, v in identity.items()):
            raise ValueError("Configuration or labels changed; start a new run.")
    else:
        if resume or rows_path.exists():
            raise ValueError("Cannot resume without a matching run_config.json.")
        revisions = {str(r): HfApi().model_info(mid).sha for r, mid in MODELS.items()}
        cfg = {**identity, "model_revisions": revisions,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(),
               "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "accelerate", "huggingface_hub", "pillow"]},
               "gpu_name": torch.cuda.get_device_name(0),
               "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    versions = {p: importlib.metadata.version(p) for p in cfg["versions"]}
    if versions != cfg["versions"] or torch.cuda.get_device_name(0) != cfg["gpu_name"]:
        raise ValueError("Runtime changed; start a new output directory instead of mixing latency measurements.")
    by_id = {d["doc_id"]: d for d in docs}
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = set()
    for row in existing:
        key = (row["resolution"], row.get("prompt_language"), row["doc_id"])
        if key in completed or row["doc_id"] not in by_id or row["resolution"] not in MODELS or row.get("prompt_language") not in PROMPTS:
            raise ValueError("Invalid or repeated existing prediction")
        doc = by_id[row["doc_id"]]
        r = row["resolution"]
        if (row["sha256"], row["true_total"], row["model_id"], row["revision"], row.get("prompt")) != (doc["sha256"], doc["true_total"], MODELS[r], cfg["model_revisions"][str(r)], PROMPTS[row["prompt_language"]]):
            raise ValueError("Existing predictions do not match this run")
        completed.add(key)
    resolutions = list(MODELS); random.Random(2026).shuffle(resolutions)
    for resolution in resolutions:
        pending = [(language, d) for language in PROMPTS for d in docs
                   if (resolution, language, d["doc_id"]) not in completed]
        if not pending:
            continue
        model_id = MODELS[resolution]; revision = cfg["model_revisions"][str(resolution)]
        print("Loading", model_id, flush=True)
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, attn_implementation="eager").to("cuda").eval()
        if model.config.vision_config.image_size != resolution:
            raise ValueError("Unexpected checkpoint image size")

        def infer(doc, language):
            with Image.open(doc["image"]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                width, height = image.size
                began = time.perf_counter()
                inputs = processor(images=image, text=PROMPTS[language], return_tensors="pt")
            if tuple(inputs["pixel_values"].shape[-2:]) != (resolution, resolution):
                raise ValueError("Unexpected processor output size")
            inputs = {k: v.to("cuda", dtype=torch.float16) if v.is_floating_point() else v.to("cuda") for k, v in inputs.items()}
            prefix = inputs["input_ids"].shape[-1]
            torch.cuda.synchronize(); ready = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            torch.cuda.synchronize(); finished = time.perf_counter()
            answer = processor.decode(generated[0, prefix:], skip_special_tokens=True).strip()
            metrics = {"generation_seconds": finished-ready, "preprocess_seconds": ready-began,
                       "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated()/2**30,
                       "generated_tokens": int(generated.shape[-1]-prefix),
                       "original_width": width, "original_height": height}
            del inputs, generated
            return answer, metrics

        for language in PROMPTS:
            infer(docs[0], language)  # warm-up per language; excluded from scoring/timing
        order = pending.copy(); random.Random(2026).shuffle(order)
        try:
            with rows_path.open("a", encoding="utf-8") as f:
                for language, doc in order:
                    raw, metrics = infer(doc, language)
                    row = {**{k: v for k, v in doc.items() if k != "image"},
                           "resolution": resolution, "prompt_language": language,
                           "prompt": PROMPTS[language], "model_id": model_id, "revision": revision,
                           "raw_answer": raw, **score_answer(raw, doc["true_total"]), **metrics}
                    f.write(json.dumps(row, ensure_ascii=False)+"\n"); f.flush()
                    print(resolution, language, doc["doc_id"], row["error_type"], flush=True)
        finally:
            del model, processor, infer
            gc.collect(); torch.cuda.empty_cache()
    rows = read_jsonl(rows_path)
    if len(rows) != len(MODELS)*len(PROMPTS)*len(docs):
        raise ValueError("Incomplete run; keep partial records, do not report headline results.")
    print("Complete:", len(docs), "documents × 2 models × 2 question languages. Run analysis next.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.manifest, args.images, args.out, args.resume)


if __name__ == "__main__":
    main()
