"""Usage: python infer.py "My Pro plan was charged twice, this is urgent!" """
import sys, json
from transformers import AutoTokenizer, T5ForConditionalGeneration
from common import device, predict
dev = device()
tok = AutoTokenizer.from_pretrained("outputs/model")
model = T5ForConditionalGeneration.from_pretrained("outputs/model").to(dev)
text = sys.argv[1] if len(sys.argv) > 1 else "My laptop order is delayed by a week. Please look into it."
print(json.dumps(predict(model, tok, [text], dev)[0]))
