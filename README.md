# Open-Source SFT of a Small LLM for Support-Ticket Triage

**Course:** Large Language Model (UNAIL411) | **TAE-II** | **Student:** Harsh Hedaoo, Roll 54, Sec A, Sem VII, AI
**Based on TAE-I:** Study of Tools in LLM: OpenAI Fine-Tuning API

## Problem
OpenAI is winding down its self-serve Fine-Tuning API (announced 7 May 2026). This project
reproduces the same SFT workflow with an open-source model (`google/flan-t5-small`) and
Hugging Face tools, so it keeps working regardless of one provider's platform decisions.

**Task:** read a customer support ticket and output JSON, e.g.
`{"category": "billing", "priority": "high"}`.

## Link to TAE-I
| TAE-I concept | This project |
|---|---|
| SFT with input/output pairs | 480 training examples, full SFT of flan-t5-small |
| Chat-style JSONL (messages) | `data/*.jsonl` uses the exact OpenAI chat format |
| Training metrics (loss) | train loss + validation exact-match printed each epoch |
| Base model vs fine-tuned | `evaluate.py` compares both on a held-out test set |
| Task-specific, not broad benchmark gains | Evaluated on task accuracy only |

## Files
- `make_data.py` - builds 600 synthetic tickets (train 480 / val 60 / test 60)
- `common.py` - data loading, prompt format, prediction helpers
- `train.py` - SFT loop (AdamW, lr 3e-4, 5 epochs, batch 16)
- `evaluate.py` - base vs fine-tuned: category acc, priority acc, exact match
- `infer.py` - try the model on any ticket

## How to run (laptop or Google Colab)
```bash
pip install -r requirements.txt
python make_data.py
python train.py        # a few minutes on CPU/GPU
python evaluate.py     # writes outputs/results.json
python infer.py "My Pro plan was charged twice, this is urgent!"
```

## Results
Run `evaluate.py` and paste the numbers here:

| Model | Category acc | Priority acc | Exact match |
|---|---|---|---|
| Base flan-t5-small | _fill_ | _fill_ | _fill_ |
| Fine-tuned | _fill_ | _fill_ | _fill_ |

## Notes and limitations
- The dataset is synthetic and template-based, so high accuracy is expected; real tickets
  would be messier. Honest framing for viva: this demonstrates the pipeline, not production quality.
- T5's vocabulary lacks curly braces, so the model is trained on `category: x | priority: y`
  and the output is converted to JSON in `common.parse_output`.
- Priority is determined by tone phrases (urgent/no rush), a deliberately simple rule.
- Possible extensions: LoRA/PEFT, a larger model (Qwen2.5-0.5B), real ticket data, DPO.
