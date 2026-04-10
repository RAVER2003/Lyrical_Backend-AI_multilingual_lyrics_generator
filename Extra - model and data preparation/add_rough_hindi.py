"""
add_rough_hindi.py  ─  Kaggle edition
───────────────────────────────────────
Translates the English 'literal' field → rough Hindi using IndicTrans2
(ai4bharat/indictrans2-en-indic-1B) and saves it as a new 'rough_hindi' field.

This is Stage 1 training data prep for the two-stage pipeline:
    English → [IndicTrans2 en→hi] → rough Hindi → [IndicBART] → polished lyric

HOW TO USE ON KAGGLE
────────────────────
1. Create a new Kaggle notebook (GPU T4 x2, internet ON).
2. Upload training_data.jsonl as a Kaggle Dataset.
3. Run Cell 1, then RESTART THE KERNEL, then run Cell 2 onward.
4. Download training_data.jsonl from the Output tab when done.

Cell 1 – install dependencies  ← restart kernel after this
──────────────────────────────
!pip install -q "transformers==4.38.2" sentencepiece sacremoses
!pip install -q git+https://github.com/VarunGumma/IndicTransToolkit
"""

# ── Cell 2 – imports & config ────────────────────────────────────────────────
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# ── EDIT THIS PATH to match your Kaggle dataset mount ────────────────────────
INPUT  = '/kaggle/input/<your-dataset-name>/training_data.jsonl'
OUTPUT = '/kaggle/working/training_data.jsonl'
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 4
MODEL_NAME = 'ai4bharat/indictrans2-en-indic-1B'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Device : {DEVICE}')
print(f'GPUs available: {torch.cuda.device_count()}')

# ── Cell 3 – load model ──────────────────────────────────────────────────────
# device_map="auto" spreads the model across all available GPUs automatically
print(f'Loading {MODEL_NAME}...')
ip        = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                   device_map='auto')
model.eval()
print('Model loaded.\n')


# ── Cell 4 – translate function (English → Hindi) ────────────────────────────
def translate_batch(texts):
    batch  = ip.preprocess_batch(texts, src_lang='eng_Latn', tgt_lang='hin_Deva')
    # Use the device of the model's first parameter (works with device_map='auto')
    input_device = next(model.parameters()).device
    inputs = tokenizer(batch, return_tensors='pt', padding=True,
                       truncation=True, max_length=256).to(input_device)
    with torch.no_grad():
        out = model.generate(**inputs, num_beams=4, max_length=256)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    return ip.postprocess_batch(decoded, lang='hin_Deva')


# ── Cell 5 – load records ────────────────────────────────────────────────────
records = []
with open(INPUT, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f'Loaded {len(records)} records. Translating in batches of {BATCH_SIZE}...\n')

# ── Cell 6 – translate ───────────────────────────────────────────────────────
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i : i + BATCH_SIZE]
    texts = [r['literal'] for r in batch]   # English input
    try:
        rough_hindi = translate_batch(texts)
        for record, rh in zip(batch, rough_hindi):
            record['rough_hindi'] = rh
    except Exception as e:
        print(f'  Batch {i // BATCH_SIZE + 1} failed ({e}). Copying literal as fallback.')
        for record in batch:
            record['rough_hindi'] = record.get('literal', '')

    done = min(i + BATCH_SIZE, len(records))
    if done % 200 == 0 or done == len(records):
        print(f'  {done}/{len(records)} records done...')

# ── Cell 7 – save ────────────────────────────────────────────────────────────
with open(OUTPUT, 'w', encoding='utf-8') as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f'\nDone!  {len(records)} records with rough_hindi  →  {OUTPUT}')
print('\nField layout per record:')
print('  literal     = English meaning')
print('  rough_hindi = flat Hindi from IndicTrans2  ← Stage 2 input')
print('  output      = polished Hindi lyric          ← Stage 2 target')
