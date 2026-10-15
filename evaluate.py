"""Compare the base model (no fine-tuning) with the fine-tuned model on the test set."""
import json, sys
from transformers import AutoTokenizer, T5ForConditionalGeneration
from common import load_jsonl, device, predict

dev = device(); test = load_jsonl("data/test.jsonl")
texts, gold = [t for t, _ in test], [l for _, l in test]
results = {}
for name, path in [("base_flan-t5-small", "google/flan-t5-small"), ("fine_tuned", "outputs/model")]:
    tok = AutoTokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path).to(dev)
    preds = predict(model, tok, texts, dev); n = len(gold)
    results[name] = {
        "category_acc": sum(p["category"] == g["category"] for p, g in zip(preds, gold)) / n,
        "priority_acc": sum(p["priority"] == g["priority"] for p, g in zip(preds, gold)) / n,
        "exact_match":  sum(p == g for p, g in zip(preds, gold)) / n}
    print(name, json.dumps(results[name], indent=2))
json.dump(results, open("outputs/results.json", "w"), indent=2)
