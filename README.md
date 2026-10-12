# PaliGemma 2 on Turkish Receipts

**A paired evaluation of total-amount extraction with 224px and 448px model checkpoints.**

Author: Tuğçe Karakuş

**Status: implementation ready; real experiment not yet executed.** This repository contains no fabricated benchmark results. It evaluates **30–50 independently collected Turkish receipts**; target size is 50. The first study uses receipts only, not a mixture of invoices, delivery notes and receipts.

The research question is whether the 448px checkpoint improves exact total extraction enough to justify its measured GPU resource demands. Human correction costs are a separate workflow question. Both models see the same documents and frozen question.

| Component | Design |
|---|---|
| Models | `google/paligemma2-3b-mix-224`, `google/paligemma2-3b-mix-448` |
| Ground truth | Human-verified final total payable, entered before inference |
| Primary metric | Exact monetary total match; unreadable/format failures count as failures |
| Secondary measures | Numerical errors, format errors, generation latency, allocated GPU memory |
| Comparison | Paired document outcomes; Wilson intervals; paired bootstrap and exploratory exact McNemar |
| Economics | Measured model errors + explicitly hypothetical human-time scenario |

## Start in Colab

[Open the experiment in Colab](https://colab.research.google.com/github/karakustugce/paligemma2-turkish-receipts/blob/main/notebooks/colab_evaluation.ipynb) and choose a T4 GPU runtime. The notebook downloads this repository, then asks for receipt images and the completed manifest. Log in to Hugging Face, run the evaluation and export the results. Both model pages require acceptance of Google's access terms. GPU availability and elapsed runtime depend on Colab.

## Run locally on a CUDA GPU

Use Python 3.11+ with a compatible CUDA PyTorch installation:

```bash
python -m pip install -r requirements.txt
hf auth login
python -m unittest discover -s tests -v
python -m pg2receipts.run --manifest data/manifest.csv --images data/images --out results/run01
python -m pg2receipts.analyze --run results/run01
```

An interrupted run can resume with the same files and runtime:

```bash
python -m pg2receipts.run --manifest data/manifest.csv --images data/images --out results/run01 --resume
```

Configuration drift, changed labels, duplicate images, incomplete paired runs and mismatched scoring records are rejected. A new prompt or dataset requires a new output directory. Model revision SHAs, dependency versions, image hashes and the GPU are recorded.

## Prepare the data

Copy `data/manifest_template.csv` to `data/manifest.csv`, fill in the true totals and place the corresponding images under `data/images/`. Decimal commas must be quoted in CSV; using a decimal point avoids CSV separator ambiguity. `1234.56` means 1,234.56 TRY. Keep the entire receipt in the image; the official processor handles resizing. Do not manually crop one model's input differently.

Suggested labels: `capture_quality` = `clear` or `difficult`; `length_group` = `short` or `long`; `merchant_group` = an anonymous group ID such as `m01`. Freeze these labels before viewing model outputs. Do not count multiple photographs of the same receipt as independent documents.

Read [the Turkish data and experiment guide](docs/VERI_VE_DENEY.md) before collection.

## Outputs after real execution

- `run_config.json`: frozen protocol, ground truth and software/model identities.
- `predictions.jsonl` / `.csv`: raw model answers, scores and timings.
- `summary.json` / `.csv`: accuracy, confidence intervals and resource measurements.
- `subgroups.csv`: exploratory counts for clarity and receipt length, always with denominators.
- `accuracy_and_resources.png` / `.pdf`: measured accuracy and GPU resource charts.
- `economic_scenario.png` / `.pdf`: model error shares on an assumed human-time curve.
- `report.md`: generated findings with limitations, ready for manual review.

The strict scoring rule accepts a single amount, optional TL/TRY/₺, decimal comma or point, and unambiguous grouped thousands. Plain integers are accepted as zero-cent amounts. Extra prose, multiple amounts and ambiguous separators are failures. This measures success under the specified output contract, not just whether a number appears somewhere in the answer. Numerical absolute error excludes unparseable answers; it must never replace the main accuracy metric.

## Scope and publication

This is a checkpoint comparison, not an experiment isolating resolution causally: mix checkpoints differ in their training. It does not replicate the paper's task-specific DocVQA evaluation. The images are Turkish; the fixed question is English. Results from a small convenience sample do not establish performance across all Turkish documents. Repeated store templates weaken independent-sample assumptions.

GPU generation time is not human labor time or total end-to-end business cost. Training compute from the original paper is not an inference price. Human-time parameters in the economic chart are assumptions until workflow measurements exist.

Raw images and labels are excluded from Git by default. Publish code and reviewed aggregate findings; only include document examples when you have permission and have removed private information. The project does not redistribute Google model weights. No license grant is asserted for third-party documents or model checkpoints.

## Clone this repository

```bash
git clone https://github.com/karakustugce/paligemma2-turkish-receipts.git
cd paligemma2-turkish-receipts
```

Follow the Colab or local GPU instructions above. After the real experiment, update the status and publish reviewed aggregate figures/report with the recorded run configuration.

## Sources

- [PaliGemma 2 paper](https://arxiv.org/abs/2412.03555)
- [Official mix-224 checkpoint](https://huggingface.co/google/paligemma2-3b-mix-224)
- [Official mix-448 checkpoint](https://huggingface.co/google/paligemma2-3b-mix-448)
- [Transformers 4.57.1 PaliGemma documentation](https://huggingface.co/docs/transformers/v4.57.1/model_doc/paligemma)
