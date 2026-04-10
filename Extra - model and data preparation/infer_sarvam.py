"""
infer_sarvam.py
───────────────
Loads the fine-tuned LoRA adapter + base Sarvam-2b and generates
polished genre-aware rhyming Hindi lyrics from rough Hindi input.

Usage:
  python infer_sarvam.py                     # interactive prompt
  python infer_sarvam.py --file data.jsonl   # batch over a JSONL file
"""

import argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_MODEL    = 'sarvamai/sarvam-2b-v0.5'
ADAPTER_DIR   = r'e:\indic\trained model'
MAX_SEQ_LEN   = 512

# ── Genre descriptions (must match training) ─────────────────────────────────
GENRE_STYLE = {
    'devotional': 'भक्ति और आस्था से भरी',
    'romantic'  : 'प्रेम और रोमांस से भरी',
    'sad'       : 'दुख और विरह से भरी',
    'happy'     : 'खुशी और उल्लास से भरी',
    'party'     : 'उत्साह और जोश से भरी',
}

RESPONSE_TEMPLATE = '### Response:\n'

# ── Load tokenizer ────────────────────────────────────────────────────────────
print('Loading tokenizer…')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# ── Load model in 4-bit + LoRA adapter ───────────────────────────────────────
print(f'Loading base model {BASE_MODEL} in 4-bit…')
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = 'nf4',
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = bnb_cfg,
    device_map          = 'auto',
    torch_dtype         = torch.float16,
)
print(f'Applying LoRA adapter from {ADAPTER_DIR}…')
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
print('Model ready.\n')

device = next(model.parameters()).device

# ── Prompt builder (identical to training) ───────────────────────────────────
def build_prompt(rec):
    genre      = rec.get('genre', 'hindi')
    style_desc = GENRE_STYLE.get(genre, genre)
    rhyme_sch  = rec.get('rhyme_scheme', '?')
    rhyme_grp  = rec.get('rhyme_group', '?')
    prev_end   = rec.get('prev_end', 'none')
    rough      = rec.get('rough_hindi', '').strip()

    prev_end_line = (
        f'पिछली पंक्ति का अंत: "{prev_end}"'
        if prev_end and str(prev_end).lower() != 'none'
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

# ── Generate ──────────────────────────────────────────────────────────────────
def generate(rec):
    prompt = build_prompt(rec)
    inputs = tokenizer(prompt, return_tensors='pt',
                       truncation=True, max_length=MAX_SEQ_LEN).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens       = 80,
            num_beams            = 4,
            no_repeat_ngram_size = 3,
            early_stopping       = True,
            pad_token_id         = tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── Entrypoints ───────────────────────────────────────────────────────────────
def run_batch(jsonl_path):
    with open(jsonl_path, encoding='utf-8') as f:
        records = [json.loads(l) for l in f if l.strip()]
    for i, rec in enumerate(records):
        result = generate(rec)
        print(f'[{i+1}/{len(records)}]')
        print(f'  GENRE    : {rec.get("genre")}  | rhyme_group: {rec.get("rhyme_group")}  | prev_end: {rec.get("prev_end")}')
        print(f'  ROUGH    : {rec.get("rough_hindi","").strip()[:100]}')
        if rec.get('output'):
            print(f'  TARGET   : {rec["output"].strip()[:100]}')
        print(f'  GENERATED: {result[:100]}')
        print()

GENRES = ['devotional', 'romantic', 'sad', 'happy', 'party']

RHYME_SCHEMES = ['AABB', 'ABAB', 'AABBA', 'AAAA', 'ABCB', 'other']

def pick(label, options, default=0):
    """Print a numbered menu and return the chosen string."""
    print(f'\n{label}')
    for i, opt in enumerate(options, 1):
        marker = ' (default)' if i - 1 == default else ''
        print(f'  {i}. {opt}{marker}')
    while True:
        raw = input(f'Enter number [1-{len(options)}] or press Enter for default: ').strip()
        if raw == '':
            return options[default]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            chosen = options[int(raw) - 1]
            if chosen == 'other':
                return input('  Type custom value: ').strip()
            return chosen
        print(f'  Please enter a number between 1 and {len(options)}.')

def run_interactive():
    print('── Hindi Lyrics Generator ──────────────────────────────')
    print('Press Ctrl+C at any time to quit.\n')
    while True:
        try:
            genre        = pick('Genre:', GENRES, default=0)
            rhyme_scheme = pick('Rhyme scheme:', RHYME_SCHEMES, default=0)

            print('\nRhyme group — which group does this line belong to?')
            print('  (e.g. for AABB: lines 1 & 2 are group A, lines 3 & 4 are group B)')
            rhyme_group = input('  Rhyme group letter [default A]: ').strip() or 'A'

            print('\nPrevious line ending — last word/syllable of the preceding line.')
            prev_end = input('  (press Enter if this is the first line): ').strip() or 'none'

            print('\nRough Hindi — the flat translation to poeticise:')
            rough_hindi = input('  > ').strip()
            if not rough_hindi:
                print('  (no input given, skipping)\n')
                continue

            rec = {
                'genre'       : genre,
                'rhyme_scheme': rhyme_scheme,
                'rhyme_group' : rhyme_group,
                'prev_end'    : prev_end,
                'rough_hindi' : rough_hindi,
            }

            print('\nGenerating…')
            result = generate(rec)
            print(f'\n  ✦ {result}\n')
            print('─' * 55)

        except KeyboardInterrupt:
            print('\nDone.')
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Path to a JSONL file for batch inference')
    args = parser.parse_args()

    if args.file:
        run_batch(args.file)
    else:
        run_interactive()
