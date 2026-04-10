"""
add_literals.py  ─  Kaggle edition
────────────────────────────────────
Fills the empty 'literal' field in training_data.jsonl with English
translations using IndicTrans2 (ai4bharat/indictrans2-indic-en-1B).

HOW TO USE ON KAGGLE
────────────────────
1. Create a new Kaggle notebook (GPU T4 x2, internet ON).
2. Upload your training_data.jsonl as a Kaggle Dataset and add it to the
   notebook.  The file will appear at:
       /kaggle/input/<your-dataset-name>/training_data.jsonl
   Update INPUT below to match that path.
3. Run Cell 1 first to install dependencies, then run Cell 2 onward.
4. The translated file is saved to OUTPUT (inside /kaggle/working/).

Cell 1 – install dependencies  ← run this cell first, then restart the kernel
──────────────────────────────
!pip install -q "transformers==4.38.2" sentencepiece sacremoses
!pip install -q git+https://github.com/VarunGumma/IndicTransToolkit
"""

# ── Cell 2 – imports & config ────────────────────────────────────────────────
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# ── EDIT THESE TWO PATHS ─────────────────────────────────────────────────────
INPUT  = '/kaggle/input/<your-dataset-name>/training_data.jsonl'
OUTPUT = '/kaggle/working/training_data.jsonl'
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 16
MODEL_NAME = 'ai4bharat/indictrans2-indic-en-1B'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Device : {DEVICE}')

# ── Cell 3 – load model ──────────────────────────────────────────────────────
print(f'Loading {MODEL_NAME}...')
ip        = IndicProcessor(inference=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
model.eval()
print('Model loaded.\n')


# ── Cell 4 – translate function ──────────────────────────────────────────────
def translate_batch(texts):
    batch  = ip.preprocess_batch(texts, src_lang='hin_Deva', tgt_lang='eng_Latn')
    inputs = tokenizer(batch, return_tensors='pt', padding=True,
                       truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, num_beams=4, max_length=256)
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    return ip.postprocess_batch(decoded, lang='eng_Latn')


# ── Load ────────────────────────────────────────────────────────────────────
records = []
with open(INPUT, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f'Loaded {len(records)} records. Translating in batches of {BATCH_SIZE}...\n')

# ── Translate ────────────────────────────────────────────────────────────────
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i : i + BATCH_SIZE]
    texts = [r['output'] for r in batch]
    try:
        translations = translate_batch(texts)
        for record, translation in zip(batch, translations):
            record['literal'] = translation
    except Exception as e:
        print(f'  Batch {i // BATCH_SIZE + 1} failed ({e}). Leaving blank.')

    done = min(i + BATCH_SIZE, len(records))
    if done % 200 == 0 or done == len(records):
        print(f'  {done}/{len(records)} records done...')

# ── Save ─────────────────────────────────────────────────────────────────────
with open(OUTPUT, 'w', encoding='utf-8') as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f'\nDone!  {len(records)} records with English translations  →  {OUTPUT}')
