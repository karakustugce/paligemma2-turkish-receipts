"""Paired PaliGemma 2 evaluation on Turkish receipt totals."""

MODELS = {
    224: "google/paligemma2-3b-mix-224",
    448: "google/paligemma2-3b-mix-448",
}
PROMPTS = {
    "en": "answer en What is the final total amount to pay on this receipt? Give only the numeric amount with two decimal digits.",
    "tr": "answer tr Bu fişte ödenecek nihai toplam tutar nedir? Yalnızca iki ondalık basamaklı sayısal tutarı yaz.",
}
__version__ = "0.3.0"
