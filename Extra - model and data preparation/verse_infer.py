"""
verse_infer.py
──────────────
Edit the CONFIG section below, then run:
    python verse_infer.py

Generates every line in order, automatically passing each
line's last word as rhyme context (prev_end) for the next.
"""

import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ╔══════════════════════════════════════════════════════════════════╗
# ║                         CONFIG                                  ║
# ╠══════════════════════════════════════════════════════════════════╣

GENRE = 'party'      # devotional | romantic | sad | happy | party

RHYME_SCHEME = 'AABBCC'   # One letter per line position; cycles if shorter
                        # than ROUGH_LINES.
                        # Common schemes: AABB  ABAB  AABBA  AAAA  ABCB

# One rough Hindi line per element — in verse order.
ROUGH_LINES = [
    "यह हिट, वह बर्फ की ठंड",
    "मिशेल फाइफर, वह सफेद सोना",
    "यह उनके लिए हुड लड़कियां हैं",
    "वे अच्छी लड़कियाँ हैं, सीधी उत्कृष्ट कृतियाँ",
    "स्टाइलिन ', विलिन', इसे शहर में जीवंत करें",
    "सेंट लॉरेंट के साथ गपशप हुई",
    "खुद को चूमना है, मैं बहुत सुंदर हूँ"
]

# ╚══════════════════════════════════════════════════════════════════╝

BASE_MODEL  = 'sarvamai/sarvam-2b-v0.5'
ADAPTER_DIR = r'e:\indic\trained model'
MAX_SEQ_LEN = 512

GENRE_STYLE = {
    'devotional': 'भक्ति और आस्था से भरी',
    'romantic'  : 'प्रेम और रोमांस से भरी',
    'sad'       : 'दुख और विरह से भरी',
    'happy'     : 'खुशी और उल्लास से भरी',
    'party'     : 'उत्साह और जोश से भरी',
}
RESPONSE_TEMPLATE = '### Response:\n'

# ── Load tokenizer & model ───────────────────────────────────────────────────
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

print(f'Loading {BASE_MODEL} in 4-bit...')
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
    dtype               = torch.float16,
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
print('Model ready.\n')

device = next(model.parameters()).device

# ── Helpers ──────────────────────────────────────────────────────────────────
def last_word(text):
    words = re.sub(r'[।,.!?"\']', '', text).split()
    return words[-1] if words else 'none'

def build_prompt(rough, rhyme_group, prev_end):
    style_desc    = GENRE_STYLE.get(GENRE, GENRE)
    prev_end_line = (
        f'पिछली पंक्ति का अंत: "{prev_end}"'
        if prev_end and prev_end.lower() != 'none'
        else 'यह पहली पंक्ति है।'
    )
    return (
        f'### Instruction:\n'
        f'नीचे दिए गए हिंदी अनुवाद को एक {style_desc} गाने की काव्यात्मक पंक्ति में बदलें।\n'
        f'Genre: {GENRE} | Rhyme scheme: {RHYME_SCHEME} | Rhyme group: {rhyme_group}\n'
        f'{prev_end_line}\n\n'
        f'### Input:\n{rough}\n\n'
        f'{RESPONSE_TEMPLATE}'
    )

def generate_line(rough, rhyme_group, prev_end):
    prompt = build_prompt(rough, rhyme_group, prev_end)
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

# ── Generate full verse ──────────────────────────────────────────────────────
print(f'Genre: {GENRE}  |  Rhyme scheme: {RHYME_SCHEME}')
print('-' * 55)

generated_lines = []
prev_end = 'none'

for i, rough in enumerate(ROUGH_LINES):
    rg   = RHYME_SCHEME[i % len(RHYME_SCHEME)]
    line = generate_line(rough, rg, prev_end)
    generated_lines.append(line)
    prev_end = last_word(line)
    print(f'  [{i+1}|{rg}]  {line}')

print('\n-- Full verse ------------------------------------------')
for line in generated_lines:
    print(f'  {line}')
print('-' * 55)


