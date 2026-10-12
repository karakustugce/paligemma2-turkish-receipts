"""Export paired accuracy, resource measurements, and explicitly hypothetical economics."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from . import MODELS
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
    if not 30 <= len(declared) <= 50 or len(rows) != 2*len(declared):
        raise ValueError("A complete 30–50-document paired run is required")
    for row in rows:
        doc = declared.get(row["doc_id"])
        if doc is None or any(row[k] != doc[k] for k in ["sha256", "true_total"]):
            raise ValueError("Prediction does not match frozen ground truth")
        r = row["resolution"]
        if row["model_id"] != MODELS[r] or row["revision"] != cfg["model_revisions"][str(r)]:
            raise ValueError("Checkpoint mismatch")
        rescored = score_answer(row["raw_answer"], row["true_total"])
        if any(row[k] != v for k, v in rescored.items()):
            raise ValueError("Stored score differs from frozen scoring rule")
    paired = paired_stats(rows)
    summaries = []
    for r in MODELS:
        group = [x for x in rows if x["resolution"] == r]
        n = len(group); correct = sum(x["correct"] for x in group)
        lo, hi = wilson(correct, n)
        numeric = [x["absolute_error"] for x in group if x["absolute_error"] is not None]
        summaries.append({"resolution": r, "n": n, "correct": correct,
                          "accuracy": correct/n, "accuracy_wilson95_low": lo,
                          "accuracy_wilson95_high": hi,
                          "wrong_numeric_total": sum(x["error_type"] == "wrong_total" for x in group),
                          "format_or_unreadable": sum(x["error_type"] == "format_or_unreadable" for x in group),
                          "mean_absolute_error_numeric_answers_only": statistics.mean(numeric) if numeric else None,
                          "median_generation_seconds": statistics.median(x["generation_seconds"] for x in group),
                          "median_preprocess_seconds": statistics.median(x["preprocess_seconds"] for x in group),
                          "max_gpu_allocated_gb": max(x["peak_gpu_allocated_gb"] for x in group)})
    result = {"models": summaries, "paired": paired,
              "scope": "Small convenience sample of Turkish receipts. Checkpoint comparison, not isolated causal resolution effect.",
              "human_time_status": "Not measured; scenario figures use assumed human times."}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    fieldnames = list(rows[0])
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    subgroup_rows = []
    for dimension in ["capture_quality", "length_group"]:
        for value in sorted({x[dimension] for x in rows}):
            for r in MODELS:
                group = [x for x in rows if x["resolution"] == r and x[dimension] == value]
                if group:
                    subgroup_rows.append({"dimension": dimension, "value": value, "resolution": r,
                                          "n": len(group), "correct": sum(x["correct"] for x in group)})
    with (out / "subgroups.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dimension", "value", "resolution", "n", "correct"])
        writer.writeheader(); writer.writerows(subgroup_rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    colors = ["#B5663F", "#137D80"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.0))
    for i, s in enumerate(summaries):
        acc = s["accuracy"]*100
        axes[0].bar(i, acc, color=colors[i], width=.55)
        axes[0].errorbar(i, acc, yerr=[[acc-s["accuracy_wilson95_low"]*100], [s["accuracy_wilson95_high"]*100-acc]], fmt="none", color="#333333", capsize=4)
        axes[0].text(i, 108, f'{s["correct"]}/{s["n"]}', ha="center", fontsize=10)
        axes[1].bar(i, s["median_generation_seconds"], color=colors[i], width=.55)
        axes[2].bar(i, s["max_gpu_allocated_gb"], color=colors[i], width=.55)
    for ax in axes:
        ax.set_xticks([0, 1], ["mix-224", "mix-448"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set(title="Toplam tutar doğruluğu", ylabel="Doğru cevap (%)", ylim=(0, 118))
    axes[1].set(title="Medyan GPU üretim süresi", ylabel="Saniye / belge")
    axes[2].set(title="En yüksek GPU tahsisli bellek", ylabel="GB")
    fig.suptitle(f"Aynı {paired['n']} Türkçe fişte iki kontrol noktası", fontsize=13)
    fig.text(.02, .01, "Doğruluk: Wilson %95 aralığı. Süre: ısınma sonrası tek üretim; indirme, yükleme ve insan emeği dahil değil.\nBellek: PyTorch tahsisli bellek; toplam GPU kullanımının tamamı değil.", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .93))
    fig.savefig(out / "accuracy_and_resources.png", dpi=300)
    fig.savefig(out / "accuracy_and_resources.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 4.5))
    p = np.linspace(0, 1, 201)
    hours = lambda q: 1000*(4-1-1-5*q)/60
    ax.plot(p*100, hours(p), color="#68737B", label="Varsayımsal saat eğrisi")
    ax.axhline(0, color="#68737B", lw=.7); ax.axvline(40, color="#68737B", ls="--", lw=.8)
    for i, s in enumerate(summaries):
        q = 1-s["accuracy"]
        ax.errorbar(100*q, hours(q), xerr=[[100*(s["accuracy_wilson95_high"]-s["accuracy"])], [100*(s["accuracy"]-s["accuracy_wilson95_low"])]],
                    color=colors[i], marker=["o", "s"][i], markersize=8, capsize=4,
                    label=f'mix-{s["resolution"]}: {s["n"]-s["correct"]}/{s["n"]} başarısız')
    ax.set(xlim=(-2, 102), xlabel="Ölçülen hata payı (%) - düzeltme payının vekili varsayılıyor",
           ylabel="Varsayımsal kazanılan saat / hafta", title="Hata payı ölçülür; insan süreleri varsayılır")
    ax.legend(loc="lower left"); ax.grid(axis="y", alpha=.15)
    fig.text(.02, .01, "N=1.000; dakika: mevcut=4, hazırlama=1, kontrol=1, ek düzeltme=5.\nGerçekleşmiş işçilik tasarrufu veya parasal yatırım getirisi değildir.", fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(out / "economic_scenario.png", dpi=300)
    fig.savefig(out / "economic_scenario.pdf"); plt.close(fig)

    a, b = summaries
    low, high = paired["paired_bootstrap95"]
    report = f"""# Türkçe fiş değerlendirmesi

**Ölçülmüş sonuçlar:** Aynı {paired['n']} fişte mix-224, {a['correct']} doğru; mix-448, {b['correct']} doğru toplam verdi.

| Ölçüm | mix-224 | mix-448 |
|---|---:|---:|
| Tam doğru toplam | {a['correct']}/{a['n']} | {b['correct']}/{b['n']} |
| Doğruluk | %{100*a['accuracy']:.1f} | %{100*b['accuracy']:.1f} |
| Yanlış sayısal toplam | {a['wrong_numeric_total']} | {b['wrong_numeric_total']} |
| Ayrıştırılamayan / okunamayan | {a['format_or_unreadable']} | {b['format_or_unreadable']} |
| Medyan GPU üretimi (sn) | {a['median_generation_seconds']:.3f} | {b['median_generation_seconds']:.3f} |

448 eksi 224 doğruluk farkı: {100*paired['accuracy_difference_448_minus_224']:.1f} yüzde puan.
Eşleştirilmiş bootstrap %95 aralığı: [{100*low:.1f}, {100*high:.1f}] yüzde puan.
Yalnız 448'in doğru olduğu fiş: {paired['only_448_correct']}; yalnız 224'ün doğru olduğu: {paired['only_224_correct']}.
Keşifsel exact McNemar p değeri: {paired['mcnemar_exact_p']:.4f}.

Bu küçük kolayda örneklem bütün Türkçe belgeleri temsil etmez. Birincil ölçüt toplam tutarın tam eşleşmesidir; biçim hataları başarısız sayılır. Sayısal yanıtların mutlak hata ortalaması, ayrıştırılamayan cevapları dışarıda bırakır ve genel başarı yerine kullanılmamalıdır. Alt gruplar keşifseldir; çoklu testle doğrulanmış iddialar değildir. Güven aralıkları örneklemin bağımsız belgelerden oluştuğunu varsayar; benzer işletme şablonları bu varsayımı zayıflatabilir. Mix kontrol noktası karşılaştırması, çözünürlüğün tek başına nedensel etkisini ayırmaz.

`accuracy_and_resources.png`: model doğruluğu ve GPU kaynak ölçümleri.
`economic_scenario.png`: ölçülen hata payı ile **varsayılan** insan sürelerini birleştirir. İnsan emeği veya parasal yatırım getirisi ölçülmemiştir.

Model sürümleri, görüntü hash'leri ve soru `run_config.json` içinde; ham yanıtlar `predictions.jsonl` içinde saklıdır. Test belgelerinde prompt ayarlamayın. Sonuçları yayımlamadan önce ham yanıtları ve referans tutarları kontrol edin.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print("Saved summary, paired statistics, subgroup counts, figures and report to", out)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True)
    analyze(parser.parse_args().run)


if __name__ == "__main__":
    main()
