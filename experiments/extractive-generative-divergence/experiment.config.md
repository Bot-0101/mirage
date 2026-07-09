# experiment.config.md — extractive-generative-divergence

## Identity
- **Name:** extractive-generative-divergence (confabulation signal)
- **Created:** 2026-07-07
- **Seed:** 42 (pin everywhere: `torch`, `numpy`, `random`, HF `set_seed`)
- **Status:** scaffolded — next phase `/scope`

## One-line goal
Test whether the **semantic divergence between a model's grounded/extractive answer and its
free-form generative answer** is a **label-free** confabulation/hallucination detector.

## Config (with reasoning)
| Decision | Choice | Why |
|---|---|---|
| **Library** | HuggingFace `transformers` (behavioral phase) | The discriminating test is generate-and-score; no activation hooks needed yet. Plain HF gives the most reliable chat-template handling — critical per repo's *tokenizer-first* rule. |
| **Model** | `Qwen/Qwen2.5-3B-Instruct` | Open (no gated token), strong instruction-following at 3B — both modes depend on it obeying "extract a verbatim span / abstain" vs "answer freely". TransformerLens-supported, so the mechanistic follow-up phase is cheap. |
| **Compute** | Kaggle (GPU) | Persistent `/kaggle/working` output dir; T4×2/P100 fits a 3B model in fp16 for the ~500-item slice. |
| **Same-model, two-mode** | one set of weights, two prompts | Isolates *grounding* as the only manipulated variable. Avoids the two-model confound in the published work (RoBERTa extractive vs Flan-T5 generative). |
| **Primary dataset (candidate)** | SQuAD 2.0 | Inherits prior extractive setup; its answerable/unanswerable split gives **built-in confabulation labels** (unanswerable ⇒ any confident generative answer is a confabulation by construction). To be locked in `/scope`. |
| **Seed** | 42 | Reproducibility is non-negotiable; record in every results file. |

## Escalation plan
Start small (3B, ~500 items). Escalate to a larger HF model / more data **only if** the signal
survives the discriminating test but is compute-sensitive. Reintroduce the two-model
(RoBERTa/Flan-T5) setup later as a *robustness axis*, not the core design.

## Kaggle setup snippet (ready to paste)
```python
# --- Cell 1: install (Kaggle usually has transformers; pin if needed) ---
!pip -q install -U "transformers>=4.45" accelerate datasets sentence-transformers

# --- Cell 2: imports + reproducibility ---
import os, random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = "Qwen/Qwen2.5-3B-Instruct"   # open, no HF_TOKEN required
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
model.eval()

# --- Cell 3: SANITY — print one chat-templated tokenization BEFORE trusting anything ---
msgs = [{"role":"user","content":"Context: The sky is blue.\nQuestion: What colour is the sky?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
print(repr(tok.decode(ids[0])))          # verify chat format is correct
print("n_tokens:", ids.shape[-1])

RESULTS_DIR = "/kaggle/working/results"   # persists as Kaggle output
os.makedirs(RESULTS_DIR, exist_ok=True)
```
> No `HF_TOKEN` needed (Qwen2.5 is ungated). If we later switch to Llama-3.2, set `HF_TOKEN` in Kaggle Secrets.

## Repo-rule reminders for this experiment
- **Tokenizer/chat-template first** — the same-model two-mode design lives or dies on the two
  prompts being formatted correctly. Print a decoded chat-templated sample in the first slice (Cell 3 above).
- **Baseline to beat:** semantic entropy (Farquhar/Kuhn) / SelfCheckGPT at *matched compute*. Divergence
  is only interesting if ~1 extra forward pass rivals N-sample consistency methods.
- **Read random examples**, not cherry-picked. Include random divergent/non-divergent items in any writeup.
- **"Excitement is bullshit"** — if divergence AUROC looks great, spend 5 min on "how is this an artifact?"
  (e.g. divergence just re-encoding answerability, or length/format differences between modes).
