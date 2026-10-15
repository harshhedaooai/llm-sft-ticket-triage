import json, re, torch

PREFIX = "Triage this support ticket. Give category and priority: "

def load_jsonl(path):
    """Return list of (ticket_text, label_dict) from OpenAI chat-format JSONL."""
    out = []
    for line in open(path):
        m = json.loads(line)["messages"]
        out.append((m[1]["content"], json.loads(m[2]["content"])))
    return out

def to_target(label):
    # T5's vocabulary has no curly braces, so train on a flat string and convert to JSON afterwards.
    return f"category: {label['category']} | priority: {label['priority']}"

def parse_output(text):
    c = re.search(r"category:\s*(\w+)", text); p = re.search(r"priority:\s*(\w+)", text)
    return {"category": c.group(1) if c else None, "priority": p.group(1) if p else None}

def device():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

@torch.no_grad()
def predict(model, tok, texts, dev, bs=32):
    model.eval(); preds = []
    for i in range(0, len(texts), bs):
        enc = tok([PREFIX + t for t in texts[i:i+bs]], return_tensors="pt",
                  padding=True, truncation=True, max_length=128).to(dev)
        gen = model.generate(**enc, max_new_tokens=24)
        preds += [parse_output(s) for s in tok.batch_decode(gen, skip_special_tokens=True)]
    return preds
