"""Supervised fine-tuning of flan-t5-small on ticket triage (SFT, plain PyTorch loop)."""
import random, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration
from common import load_jsonl, to_target, PREFIX, device, predict

BASE, OUT, EPOCHS, BS, LR = "google/flan-t5-small", "outputs/model", 5, 16, 3e-4
random.seed(0); torch.manual_seed(0)
dev = device(); print("device:", dev)
tok = AutoTokenizer.from_pretrained(BASE)
model = T5ForConditionalGeneration.from_pretrained(BASE).to(dev)

train, val = load_jsonl("data/train.jsonl"), load_jsonl("data/val.jsonl")

def collate(batch):
    x = tok([PREFIX + t for t, _ in batch], return_tensors="pt", padding=True, truncation=True, max_length=128)
    y = tok([to_target(l) for _, l in batch], return_tensors="pt", padding=True, truncation=True, max_length=32).input_ids
    y[y == tok.pad_token_id] = -100   # ignore padding in the loss
    x["labels"] = y
    return x

loader = DataLoader(train, batch_size=BS, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

for ep in range(1, EPOCHS + 1):
    model.train(); total = 0
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward(); opt.step(); opt.zero_grad(); total += loss.item()
    preds = predict(model, tok, [t for t, _ in val], dev)
    acc = sum(p == l for p, (_, l) in zip(preds, val)) / len(val)
    print(f"epoch {ep}/{EPOCHS}  train_loss={total/len(loader):.4f}  val_exact_match={acc:.3f}")

model.save_pretrained(OUT); tok.save_pretrained(OUT)
print("saved to", OUT)
