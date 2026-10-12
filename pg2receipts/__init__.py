"""Paired PaliGemma 2 evaluation on Turkish receipt totals."""

MODELS = {
    224: "google/paligemma2-3b-mix-224",
    448: "google/paligemma2-3b-mix-448",
}
PROMPT = (
    "answer en What is the final total amount to pay on this receipt? "
    "Give only the numeric amount with two decimal digits."
)
__version__ = "0.1.0"
