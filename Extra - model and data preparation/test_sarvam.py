"""
test_sarvam.py  ─  Kaggle edition
───────────────────────────────────
Tests sarvamai/sarvam-2b-v0.5 zero-shot for converting rough Hindi
into polished genre-aware rhyming Hindi lyrics.

HOW TO USE ON KAGGLE
────────────────────
1. Create a new Kaggle notebook (GPU T4 x2, internet ON).
2. Upload "final rough and rhyme hindi.jsonl" as a Kaggle Dataset.
3. Run Cell 1, RESTART KERNEL, then run Cell 2 onward.
4. Inspect printed outputs to gauge zero-shot quality.

Cell 1 – install dependencies  ← restart kernel after this
──────────────────────────────
!pip install -q --upgrade transformers accelerate bitsandbytes sentencepiece
"""

# ── Cell 2 – imports & config ────────────────────────────────────────────────
import json, textwrap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── EDIT THIS PATH to match your Kaggle dataset mount ────────────────────────
INPUT = '/kaggle/input/<your-dataset-name>/final rough and rhyme hindi.jsonl'
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME  = 'sarvamai/sarvam-2b-v0.5'
NUM_SAMPLES = 10
RESPONSE_TEMPLATE = '### Response:\n'

print(f'CUDA available : {torch.cuda.is_available()}')
print(f'GPUs           : {torch.cuda.device_count()}')

# ── Cell 3 – load model in 4-bit ─────────────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = 'nf4',
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
)

print(f'Loading {MODEL_NAME} in 4-bit…')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_cfg,
    device_map          = 'auto',
    torch_dtype         = torch.float16,
)
model.eval()
print('Model loaded.\n')

# ── Cell 4 – prompt builder ──────────────────────────────────────────────────
GENRE_STYLE = {
    'devotional': 'भक्ति और आस्था से भरी',
    'romantic'  : 'प्रेम और रोमांस से भरी',
    'sad'       : 'दुख और विरह से भरी',
    'happy'     : 'खुशी और उल्लास से भरी',
    'party'     : 'उत्साह और जोश से भरी',
}

def build_prompt(rec):
    genre      = rec.get('genre', 'hindi')
    style_desc = GENRE_STYLE.get(genre, genre)
    rhyme_sch  = rec.get('rhyme_scheme', '?')
    rhyme_grp  = rec.get('rhyme_group', '?')
    prev_end   = rec.get('prev_end', 'none')
    rough      = rec.get('rough_hindi', '').strip()

    prev_end_line = (
        f'पिछली पंक्ति का अंत: "{prev_end}"'
        if prev_end and prev_end.lower() != 'none'
        else 'यह पहली पंक्ति है।'
    )

    return (
        f'### Instruction:\n'
        f'नीचे दिए गए हिंदी अनुवाद को एक {style_desc} गाने की काव्यात्मक पंक्ति में बदलें।\n'
        f'Genre: {genre} | Rhyme scheme: {rhyme_sch} | Rhyme group: {rhyme_grp}\n'
        f'{prev_end_line}\n\n'
        f'### Input:\n{rough}\n\n'
        f'{RESPONSE_TEMPLATE}'
    )

# ── Cell 5 – generation helper ───────────────────────────────────────────────
def generate(rec, max_new_tokens=80):
    prompt = build_prompt(rec)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors='pt',
                       truncation=True, max_length=512).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens       = max_new_tokens,
            max_length           = None,
            num_beams            = 4,
            no_repeat_ngram_size = 3,
            early_stopping       = True,
            pad_token_id         = tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── Cell 6 – load data & run inference ───────────────────────────────────────
with open(INPUT, encoding='utf-8') as f:
    records = [json.loads(l) for l in f if l.strip()]

testable = [r for r in records
            if r.get('rough_hindi', '').strip() and r.get('output', '').strip()]

print(f'Total records   : {len(records)}')
print(f'Testable records: {len(testable)}')
print(f'Running on first {NUM_SAMPLES} samples…\n')
print('=' * 80)

for i, rec in enumerate(testable[:NUM_SAMPLES]):
    generated = generate(rec)

    print(f'[Sample {i+1}]  genre={rec.get("genre")}  rhyme_group={rec.get("rhyme_group")}  prev_end="{rec.get("prev_end")}"')
    print(f'  ROUGH    : {textwrap.shorten(rec["rough_hindi"].strip(), width=110, placeholder="…")}')
    print(f'  GENERATED: {textwrap.shorten(generated,                  width=110, placeholder="…")}')
    print(f'  TARGET   : {textwrap.shorten(rec["output"].strip(),       width=110, placeholder="…")}')
    print()

print('=' * 80)
print('Done. If zero-shot quality is poor, run finetune_sarvam.py next.')
