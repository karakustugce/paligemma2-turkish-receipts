"""Paired 2×2 evaluation: resolution and question language."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from . import MODELS, PROMPTS
from .core import paired_stats, read_jsonl, score_answer, wilson


def analyze(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    out = Path(directory)
    cfg = json.loads((out / "run_config.json").read_text())
    rows = read_jsonl(out / "predictions.jsonl")
    declared = {d["doc_id"]: d for d in cfg["documents"]}
    dataset_types = {d.get("dataset_type", "real") for d in cfg["documents"]}
    if len(dataset_types) != 1 or not dataset_types.issubset({"real", "synthetic"}):
        raise ValueError("A run must contain exactly one supported dataset type")
    dataset_type = next(iter(dataset_types))
    is_synthetic = dataset_type == "synthetic"
    if not 30 <= len(declared) <= 50 or len(rows) != 4*len(declared):
        raise ValueError("Complete 30–50-document, four-condition run required")
    if cfg.get("prompts") != PROMPTS:
        raise ValueError("Prompt configuration differs from this protocol")
    for row in rows:
        doc = declared.get(row["doc_id"])
        if doc is None or any(row[k] != doc[k] for k in ["sha256", "true_total"]):
            raise ValueError("Prediction does not match frozen ground truth")
        r, language = row["resolution"], row.get("prompt_language")
        if r not in MODELS or language not in PROMPTS or row.get("prompt") != PROMPTS[language]:
            raise ValueError("Unexpected condition or question")
        if row["model_id"] != MODELS[r] or row["revision"] != cfg["model_revisions"][str(r)]:
            raise ValueError("Checkpoint mismatch")
        rescored = score_answer(row["raw_answer"], row["true_total"])
        if any(row[k] != v for k, v in rescored.items()):
            raise ValueError("Stored score differs from frozen scoring rule")
    paired = {}
    for language in PROMPTS:
        selected = [x for x in rows if x["prompt_language"] == language]
        paired[f"resolution_{language}"] = paired_stats(selected)
    for r in MODELS:
        selected = [x for x in rows if x["resolution"] == r]
        paired[f"language_{r}"] = paired_stats(selected, condition_key="prompt_language", condition_values=("en", "tr"))
    summaries = []
    for r in MODELS:
        for language in PROMPTS:
            group = [x for x in rows if x["resolution"] == r and x["prompt_language"] == language]
            n = len(group); correct = sum(x["correct"] for x in group)
            lo, hi = wilson(correct, n)
            numeric = [x["absolute_error"] for x in group if x["absolute_error"] is not None]
            summaries.append({"resolution": r, "prompt_language": language, "n": n, "correct": correct,
                              "accuracy": correct/n, "accuracy_wilson95_low": lo,
                              "accuracy_wilson95_high": hi,
                              "wrong_numeric_total": sum(x["error_type"] == "wrong_total" for x in group),
                              "format_or_unreadable": sum(x["error_type"] == "format_or_unreadable" for x in group),
                              "mean_absolute_error_numeric_answers_only": statistics.mean(numeric) if numeric else None,
                              "median_generation_seconds": statistics.median(x["generation_seconds"] for x in group),
                              "median_preprocess_seconds": statistics.median(x["preprocess_seconds"] for x in group),
                              "max_gpu_allocated_gb": max(x["peak_gpu_allocated_gb"] for x in group)})
    n = len(declared)
    result = {
        "dataset_type": dataset_type,
        "conditions": summaries,
        "paired": paired,
        "primary_comparison": "resolution_tr",
        "scope": (
            "Programmatically rendered synthetic Turkish-style receipts; results do not estimate performance on real receipts."
            if is_synthetic else
            "Small Turkish receipt sample; comparison of checkpoints and language prompt conditions."
        ),
        "human_time_status": "Not measured; model latency is not human labor.",
        "economic_analysis_status": (
            "Omitted for synthetic data; synthetic accuracy must not be translated into real-world labor savings."
            if is_synthetic else
            "Not measured; any economic scenario uses assumed human times."
        ),
    }
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    methodology = {k: v for k, v in cfg.items() if k != "documents"}
    methodology["dataset_type"] = dataset_type
    methodology["n_documents"] = n
    (out / "methodology.json").write_text(json.dumps(methodology, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    subgroup_rows = []
    for dimension in ["capture_quality", "length_group"]:
        for value in sorted({x[dimension] for x in rows}):
            for r in MODELS:
                for language in PROMPTS:
                    group = [x for x in rows if x["resolution"] == r and x["prompt_language"] == language and x[dimension] == value]
                    if group:
                        subgroup_rows.append({"dimension": dimension, "value": value, "resolution": r,
                                              "prompt_language": language, "n": len(group), "correct": sum(x["correct"] for x in group)})
    with (out / "subgroups.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dimension", "value", "resolution", "prompt_language", "n", "correct"])
        writer.writeheader(); writer.writerows(subgroup_rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    colors = ["#B5663F", "#B5663F", "#137D80", "#137D80"]
    labels = [f'{s["resolution"]}\n{s["prompt_language"].upper()}' for s in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for i, s in enumerate(summaries):
        acc = s["accuracy"]*100; hatch = "//" if s["prompt_language"] == "tr" else ""
        axes[0].bar(i, acc, color=colors[i], width=.6, hatch=hatch)
        axes[0].errorbar(i, acc, yerr=[[acc-s["accuracy_wilson95_low"]*100], [s["accuracy_wilson95_high"]*100-acc]], fmt="none", color="#333333", capsize=4)
        axes[0].text(i, 108, f'{s["correct"]}/{s["n"]}', ha="center", fontsize=9)
        axes[1].bar(i, s["median_generation_seconds"], color=colors[i], width=.6, hatch=hatch)
        axes[2].bar(i, s["max_gpu_allocated_gb"], color=colors[i], width=.6, hatch=hatch)
    for ax in axes:
        ax.set_xticks(range(4), labels)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set(title="Toplam tutar doğruluğu", ylabel="Doğru cevap (%)", ylim=(0, 118))
    axes[1].set(title="Medyan GPU üretim süresi", ylabel="Saniye / belge")
    axes[2].set(title="En yüksek GPU tahsisli bellek", ylabel="GB")
    fig.suptitle(
        f"Sentetik Türkçe fiş pilotu: {n} örnek" if is_synthetic
        else f"Aynı {n} Türkçe fişte çözünürlük × soru dili",
        fontsize=13,
    )
    chart_note = (
        "Sentetik görseller ve programatik etiketler; gerçek fişlere genellenemez.\n"
        if is_synthetic else ""
    )
    chart_note += "EN: İngilizce soru; TR: Türkçe soru (taralı sütun). Doğruluk: Wilson %95 aralığı.\nSüre: ısınma sonrası tek üretim; indirme, yükleme ve insan emeği dahil değil. Bellek: PyTorch tahsisli bellek."
    fig.text(.02, .01, chart_note, fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .93))
    fig.savefig(out / "accuracy_and_resources.png", dpi=300)
    fig.savefig(out / "accuracy_and_resources.pdf"); plt.close(fig)

    if is_synthetic:
        for suffix in ("png", "pdf"):
            stale_chart = out / f"economic_scenario.{suffix}"
            if stale_chart.exists():
                stale_chart.unlink()
        (out / "economic_scenario_skipped.txt").write_text(
            "Omitted: synthetic model accuracy does not estimate real-world labor savings.\n",
            encoding="utf-8",
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        p = np.linspace(0, 1, 201)
        hours = lambda q: 1000*(4-1-1-5*q)/60
        ax.plot(p*100, hours(p), color="#68737B", label="Varsayımsal saat eğrisi")
        ax.axhline(0, color="#68737B", lw=.7); ax.axvline(40, color="#68737B", ls="--", lw=.8)
        for i, s in enumerate(summaries):
            q = 1-s["accuracy"]
            ax.errorbar(100*q, hours(q), xerr=[[100*(s["accuracy_wilson95_high"]-s["accuracy"])], [100*(s["accuracy"]-s["accuracy_wilson95_low"])]],
                        color=colors[i], marker="s" if s["prompt_language"] == "tr" else "o", markersize=8, capsize=4,
                        label=f'{s["resolution"]} / {s["prompt_language"].upper()}: {s["n"]-s["correct"]}/{s["n"]} başarısız')
        ax.set(xlim=(-2, 102), xlabel="Ölçülen hata payı (%) - düzeltme payının vekili varsayılıyor",
               ylabel="Varsayımsal kazanılan saat / hafta", title="Hata payı ölçülür; insan süreleri varsayılır")
        ax.legend(loc="lower left"); ax.grid(axis="y", alpha=.15)
        fig.text(.02, .01, "N=1.000; dakika: mevcut=4, hazırlama=1, kontrol=1, ek düzeltme=5.\nGerçekleşmiş işçilik tasarrufu veya parasal yatırım getirisi değildir.", fontsize=9)
        fig.tight_layout(rect=(0, .1, 1, 1))
        fig.savefig(out / "economic_scenario.png", dpi=300)
        fig.savefig(out / "economic_scenario.pdf"); plt.close(fig)

    if is_synthetic:
        report = (
            f"# Sentetik Türkçe fiş pilotu\n\n"
            f"{n} programatik olarak oluşturulmuş fiş, dört koşulda değerlendirildi.\n\n"
            "**Yorum sınırı:** Görseller ve etiketler sentetiktir. Bu sonuçlar gerçek fişlerdeki doğruluğu veya "
            "gerçek iş gücü tasarrufunu ölçmez.\n\n"
        )
    else:
        report = f"# Türkçe fiş değerlendirmesi\n\nAynı {n} farklı fiş, dört koşulda değerlendirildi.\n\n"
    report += "| Model | Soru | Doğru toplam | Doğruluk | Biçim/okunamama hatası | Medyan üretim (sn) |\n|---|---|---:|---:|---:|---:|\n"
    for s in summaries:
        report += f'| {s["resolution"]} | {s["prompt_language"].upper()} | {s["correct"]}/{s["n"]} | %{100*s["accuracy"]:.1f} | {s["format_or_unreadable"]} | {s["median_generation_seconds"]:.3f} |\n'
    report += "\nBirincil karşılaştırma: Türkçe soruda 448 eksi 224. Diğer üç karşılaştırma keşifseldir; çoklu test için düzeltme uygulanmadı.\n\n"
    for name, stats in paired.items():
        low, high = stats["paired_bootstrap95"]
        delta = stats[f'accuracy_difference_{stats["second"]}_minus_{stats["first"]}']
        report += f'- {name}: fark {100*delta:.1f} yüzde puan; eşleştirilmiş bootstrap %95 [{100*low:.1f}, {100*high:.1f}]; exact McNemar p={stats["mcnemar_exact_p"]:.4f}.\n'
    report += """
Bu küçük örneklem bütün Türkçe belgeleri temsil etmez. Aynı işletme şablonlarının tekrarı bağımsız örnek varsayımını zayıflatabilir. Başarı, sabit cevap kuralı altında tutarın tam eşleşmesidir; açıklamalı veya çoklu tutarlı cevaplar başarısız sayılır. Sayısal cevapların mutlak hata ortalaması ayrıştırılamayan cevapları dışlar; ana başarı ölçütünün yerine geçmez.

Dil koşulu soru metnini ve `answer en/tr` görev önekini birlikte değiştirir. Fark yalnızca çevirinin etkisi diye yorumlanamaz. Mix kontrol noktaları da eğitim açısından farklıdır; bu deney çözünürlüğün tek başına nedensel etkisini ayırmaz.

GPU üretim süresi insan emeği değildir; gerçek işçilik tasarrufu veya parasal yatırım getirisi ölçülmedi.

Ham cevaplar ve referans tutarlar `predictions.jsonl` içinde, tüm koşu ayarları `run_config.json` içinde saklanır. `methodology.json` kişisel belge kayıtlarını içermeyen yöntem özetidir. Yayından önce ham cevapları ve referans tutarları gözden geçirin.
"""
    if is_synthetic:
        report += "\nSentetik görseller, kurgusal mağaza şablonları ve programatik referans toplamları kullanıldı. Bu pilotun doğruluğu gerçek fişlerin, telefon fotoğraflarının veya gerçek iş akışlarının başarımına genellenemez. Ekonomik projeksiyon özellikle üretilmedi.\n"
    else:
        report += "\nEkonomik senaryodaki insan süreleri varsayımdır; ölçülmüş işçilik tasarrufu değildir.\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print("Saved four-condition analysis to", out)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True)
    analyze(parser.parse_args().run)


if __name__ == "__main__":
    main()
