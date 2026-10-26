"""Create a deterministic, explicitly synthetic Turkish receipt pilot dataset.

Usage:
    python -m pg2receipts.synthetic --out data/synthetic_pilot --count 50 --seed 2026

The images are programmatically rendered illustrations, not photographs or transaction
records. Ground-truth labels are generated from the same transaction data as each image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


MERCHANTS = [
    ("market", "Mavi Sepet Market - 01"),
    ("market", "Ege Taze Gıda - 02"),
    ("market", "Bereket Market - 03"),
    ("cafe", "Kıyı Kahve Evi - 01"),
    ("cafe", "Sade Fincan Kafe - 02"),
    ("cafe", "Kırkikindi Kahve - 03"),
    ("fuel", "Ege Yolu Akaryakıt - 01"),
    ("fuel", "Güney İstasyon - 02"),
    ("bakery", "Fırın Sokağı - 01"),
    ("bakery", "Buğday Evi - 02"),
]

PRODUCTS = {
    "market": [
        "Süt 1 L", "Yoğurt 1 kg", "Beyaz peynir", "Kaşar peyniri",
        "Yumurta 10'lu", "Türk kahvesi", "Çay 500 g", "Makarna",
        "Zeytinyağı 1 L", "Domates", "Salatalık", "Elma",
        "Tam buğday ekmeği", "Bulaşık deterjanı", "Kağıt havlu",
        "Mercimek 1 kg", "Tereyağı", "Maden suyu 6'lı",
    ],
    "cafe": [
        "Türk kahvesi", "Filtre kahve", "Latte", "Çay",
        "Limonata", "Simit", "Peynirli tost", "Muzlu kek",
        "San Sebastian dilim", "Su 500 ml", "Soğuk kahve",
        "Kaşarlı sandviç",
    ],
    "fuel": [
        "Kurşunsuz benzin 95", "Motorin", "LPG", "Cam suyu",
        "Su 1,5 L", "Kahve", "Çikolata", "Oto kokusu",
        "Lastik hava hizmeti", "Araç temizleme ürünü",
    ],
    "bakery": [
        "Ekmek", "Simit", "Açma", "Poğaça", "Su böreği",
        "Peynirli börek", "Tahinli çörek", "Kurabiye",
        "Ekşi mayalı ekmek", "Un kurabiyesi",
    ],
}

ADDRESSES = [
    "Denizli / Merkez", "Aydın / Efeler", "İzmir / Bornova",
    "Muğla / Menteşe", "Manisa / Yunusemre",
]

PAPER_COLORS = [(250, 249, 245), (247, 246, 240), (252, 251, 247)]
INK_COLORS = [(36, 38, 39), (45, 45, 43), (53, 52, 49)]


def _money(value: Decimal) -> str:
    value = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    whole, cents = divmod(int(value * 100), 100)
    whole_text = f"{whole:,}".replace(",", ".")
    return f"{whole_text},{cents:02d}"


def _amount(rng: random.Random, sector: str) -> Decimal:
    limits = {
        "market": (Decimal("8.50"), Decimal("479.90")),
        "cafe": (Decimal("18.00"), Decimal("189.00")),
        "fuel": (Decimal("15.00"), Decimal("950.00")),
        "bakery": (Decimal("7.50"), Decimal("139.00")),
    }
    lo, hi = limits[sector]
    cents = rng.randint(int(lo * 100), int(hi * 100))
    return Decimal(cents) / 100


def _font_paths():
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/freefont/FreeSans.ttf",
         "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).is_file() and Path(bold).is_file():
            return regular, bold
    raise RuntimeError(
        "A Turkish-capable TrueType font was not found. On Colab, install "
        "fonts-dejavu-core, then rerun this cell."
    )


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False):
    regular_path, bold_path = _font_paths()
    return ImageFont.truetype(bold_path if bold else regular_path, size=size)


def _build_transaction(rng: random.Random, sector: str, line_count: int):
    product_names = PRODUCTS[sector]
    items = []
    for _ in range(line_count):
        name = rng.choice(product_names)
        # The amount is the line total, so the printed subtotal and final total
        # can be derived without a separate label-annotation step.
        items.append((name, _amount(rng, sector)))
    subtotal = sum((amount for _, amount in items), Decimal("0.00"))
    discount_rate = rng.choice([0, 0, 0, 5, 10])
    discount = (subtotal * discount_rate / 100).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    total = (subtotal - discount).quantize(Decimal("0.01"))
    return items, subtotal, discount, total


def _entry_height(role: str) -> int:
    return {
        "brand": 40, "title": 31, "center": 27, "left": 28,
        "item": 29, "total": 40, "foot": 24, "rule": 18, "space": 14,
    }[role]


def _render(entries, width: int, rng: random.Random, degraded: bool) -> Image.Image:
    pad = 34
    height = 42 + sum(_entry_height(role) for role, _, _ in entries) + 34
    paper = rng.choice(PAPER_COLORS)
    ink = rng.choice(INK_COLORS)
    image = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(image)
    y = 30

    for role, left, right in entries:
        if role == "space":
            y += _entry_height(role)
            continue
        if role == "rule":
            draw.line((pad, y + 7, width - pad, y + 7), fill=(155, 153, 146), width=1)
        else:
            sizes = {
                "brand": 24, "title": 17, "center": 16, "left": 16,
                "item": 16, "total": 21, "foot": 13,
            }
            font = _font(sizes[role], bold=role in {"brand", "title", "total"})
            color = ink if role != "foot" else (90, 89, 84)
            if role in {"brand", "title", "center"}:
                text_width = draw.textbbox((0, 0), left, font=font)[2]
                draw.text(((width - text_width) / 2, y), left, font=font, fill=color)
            elif role == "item" and right:
                right_width = draw.textbbox((0, 0), right, font=font)[2]
                max_left_width = width - 2 * pad - right_width - 18
                clipped = left
                while clipped and draw.textbbox((0, 0), clipped, font=font)[2] > max_left_width:
                    clipped = clipped[:-2] + "…"
                draw.text((pad, y), clipped, font=font, fill=color)
                draw.text((width - pad - right_width, y), right, font=font, fill=color)
            else:
                draw.text((pad, y), left, font=font, fill=color)
                if right:
                    right_width = draw.textbbox((0, 0), right, font=font)[2]
                    draw.text((width - pad - right_width, y), right, font=font, fill=color)
        y += _entry_height(role)

    if degraded:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.86, 0.95))
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.25, 0.65)))
    angle = rng.uniform(-1.8, 1.8) if degraded else rng.uniform(-0.25, 0.25)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=paper)
    if degraded:
        noise = Image.effect_noise(image.size, 7).convert("RGB")
        image = Image.blend(image, noise, 0.025)
    return image


def _receipt_entries(rng: random.Random, sector: str, merchant: str,
                    line_count: int, doc_number: int):
    items, subtotal, discount, total = _build_transaction(rng, sector, line_count)
    sale_type = {
        "market": "PERAKENDE SATIŞ FİŞİ",
        "cafe": "YİYECEK İÇECEK FİŞİ",
        "fuel": "AKARYAKIT SATIŞ FİŞİ",
        "bakery": "SATIŞ FİŞİ",
    }[sector]
    day = date(2025, 1, 1) + timedelta(days=rng.randint(0, 620))
    entries = [
        ("brand", merchant.upper(), None),
        ("title", sale_type, None),
        ("center", rng.choice(ADDRESSES), None),
        ("left", f"Tarih: {day.strftime('%d.%m.%Y')}  Saat: {rng.randint(8, 22):02d}:{rng.randint(0, 59):02d}", None),
        ("left", f"Fiş No: {doc_number:04d}-{rng.randint(10, 99)}", None),
        ("rule", "", None),
    ]
    for name, amount in items:
        entries.append(("item", name, _money(amount)))
    entries.extend([
        ("rule", "", None),
        ("left", "Ara toplam", f"{_money(subtotal)} TL"),
    ])
    if discount:
        entries.append(("left", "İndirim", f"-{_money(discount)} TL"))
    total_label = rng.choice(["TOPLAM", "GENEL TOPLAM", "ÖDENECEK TUTAR"])
    entries.append(("total", total_label, f"{_money(total)} TL"))

    payment_mode = rng.choice(["cash", "card", "card", "none"])
    if payment_mode == "cash":
        paid_cents = ((int(total * 100) + 4999) // 5000) * 5000
        paid = Decimal(paid_cents) / 100
        change = paid - total
        entries.append(("left", "Nakit alınan", f"{_money(paid)} TL"))
        entries.append(("left", "Para üstü", f"{_money(change)} TL"))
    elif payment_mode == "card":
        entries.append(("left", "Kart ile ödeme", f"{_money(total)} TL"))
    entries.extend([
        ("space", "", None),
        ("foot", "KDV fiyata dahildir.", None),
        ("foot", "İyi günler dileriz.", None),
    ])
    return entries, total


def generate(output: str | Path, count: int = 50, seed: int = 2026) -> Path:
    if not 30 <= count <= 50:
        raise ValueError("Synthetic pilot count must be between 30 and 50.")
    rng = random.Random(seed)
    out = Path(output)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    lengths = ["short"] * (count // 2) + ["long"] * (count - count // 2)
    qualities = ["clean_render"] * ((count * 3) // 5)
    qualities += ["synthetically_degraded"] * (count - len(qualities))
    rng.shuffle(lengths)
    rng.shuffle(qualities)

    rows = []
    sector_counts = Counter()
    for index in range(count):
        sector, merchant = MERCHANTS[index % len(MERCHANTS)]
        sector_counts[sector] += 1
        length_group = lengths[index]
        line_count = rng.randint(2, 5) if length_group == "short" else rng.randint(6, 9)
        degraded = qualities[index] == "synthetically_degraded"
        entries, total = _receipt_entries(rng, sector, merchant, line_count, index + 1)
        width = rng.choice([640, 680, 720])
        image = _render(entries, width, rng, degraded)

        doc_id = f"fis_{index + 1:03d}"
        image_path = images_dir / f"{doc_id}.png"
        image.save(image_path, format="PNG", optimize=True)
        rows.append({
            "doc_id": doc_id,
            "image_path": f"{doc_id}.png",
            "true_total": f"{total:.2f}",
            "capture_quality": qualities[index],
            "length_group": length_group,
            "merchant_group": f"{sector}_{(index % len(MERCHANTS)) + 1:02d}",
            "dataset_type": "synthetic",
            "annotation_status": "synthetic_ground_truth",
        })

    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    info = {
        "dataset_type": "synthetic",
        "description": "Programmatically rendered Turkish-style receipt illustrations; not photographs or transaction records.",
        "count": count,
        "seed": seed,
        "sector_counts": dict(sorted(sector_counts.items())),
        "ground_truth": "Generated from the same transaction records used to render each image.",
        "quality_note": "clean_render and synthetically_degraded describe rendering conditions, not real camera captures.",
        "image_sha256": {
            row["doc_id"]: hashlib.sha256((images_dir / row["image_path"]).read_bytes()).hexdigest()
            for row in rows
        },
    }
    (out / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated {count} synthetic receipt images and exact labels in {out}")
    print("No real receipts, people, payment cards, or transaction records are used.")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory for manifest and images")
    parser.add_argument("--count", type=int, default=50, help="Number of synthetic receipts (30–50)")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic renderer seed")
    args = parser.parse_args()
    generate(args.out, count=args.count, seed=args.seed)


if __name__ == "__main__":
    main()
