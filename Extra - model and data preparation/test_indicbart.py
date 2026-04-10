"""
test_indicbart.py  ─  Kaggle edition
──────────────────────────────────────
Tests ai4bharat/IndicBART (zero-shot) for converting flat Hindi (rough_hindi)
into polished / rhyming Hindi lyrics (output).

HOW TO USE ON KAGGLE
────────────────────
1. Create a new Kaggle notebook (GPU T4 x2, internet ON).
2. Upload "final rough and rhyme hindi.jsonl" as a Kaggle Dataset.
3. Run Cell 1, RESTART KERNEL, then run Cell 2 onward.
4. Inspect printed outputs to gauge zero-shot quality.

Cell 1 – install dependencies  ← restart kernel after this
──────────────────────────────
!pip install -q "transformers==4.38.2" sentencepiece sacremoses
"""

# ── Cell 2 – imports & config ────────────────────────────────────────────────
import json, textwrap
import torch
from transformers import MBartForConditionalGeneration, AutoTokenizer

# ── EDIT THIS PATH to match your Kaggle dataset mount ────────────────────────
INPUT = '/kaggle/input/<your-dataset-name>/final rough and rhyme hindi.jsonl'
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME   = 'ai4bharat/IndicBART'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# IndicBART language tag for Hindi (Devanagari)
HINDI_TAG    = '<2hi>'

# How many sample records to test
NUM_SAMPLES  = 10

print(f'Device : {DEVICE}')
print(f'GPUs   : {torch.cuda.device_count()}')

# ── Cell 3 – load model ──────────────────────────────────────────────────────
print(f'Loading {MODEL_NAME}…')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model     = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
model     = model.to(DEVICE)
model.eval()
print('Model loaded.\n')

# ── Cell 4 – generation helper ───────────────────────────────────────────────
def polish_hindi(rough_text: str,
                 num_beams: int = 5,
                 max_new_tokens: int = 256,
                 length_penalty: float = 1.2) -> str:
    """
    Feed rough Hindi into IndicBART and get a more fluent / poetic output.
    IndicBART is a denoising / seq2seq model trained on Indic text.
    We set forced_bos_token_id to the Hindi tag so it generates in Hindi.
    """
    # Prefix with the language tag as IndicBART expects
    src = HINDI_TAG + ' ' + rough_text.strip()

    inputs = tokenizer(
        src,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)
    inputs.pop('token_type_ids', None)  # IndicBART doesn't use token_type_ids

    forced_bos = tokenizer.convert_tokens_to_ids(HINDI_TAG)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

# ── Cell 5 – load data & run inference ───────────────────────────────────────
with open(INPUT, encoding='utf-8') as f:
    records = [json.loads(l) for l in f if l.strip()]

# Filter to records that have both fields
testable = [r for r in records if r.get('rough_hindi', '').strip() and r.get('output', '').strip()]
print(f'Total records   : {len(records)}')
print(f'Testable records: {len(testable)}')
print(f'Running on first {NUM_SAMPLES} samples…\n')
print('=' * 80)

for i, rec in enumerate(testable[:NUM_SAMPLES]):
    rough  = rec['rough_hindi'].strip()
    target = rec['output'].strip()

    generated = polish_hindi(rough)

    print(f'[Sample {i+1}]')
    print(f'  ROUGH HINDI  : {textwrap.shorten(rough,     width=120, placeholder="…")}')
    print(f'  GENERATED    : {textwrap.shorten(generated, width=120, placeholder="…")}')
    print(f'  TARGET LYRIC : {textwrap.shorten(target,    width=120, placeholder="…")}')
    print()

print('=' * 80)
print('Done. Compare GENERATED vs TARGET LYRIC to judge zero-shot quality.')
print('If quality is poor, fine-tuning on rough_hindi → output pairs is the next step.')
