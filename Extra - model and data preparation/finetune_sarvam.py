"""
finetune_sarvam.py  ─  Kaggle edition
──────────────────────────────────────
QLoRA fine-tunes sarvamai/sarvam-2b-v0.5 to convert rough Hindi
into polished genre-aware rhyming Hindi lyrics.

Uses genre, rhyme_scheme, rhyme_group, prev_end, and rough_hindi
as the instruction prompt; output is the target completion.

HOW TO USE ON KAGGLE
────────────────────
1. Create a new Kaggle notebook (GPU T4 x2, internet ON).
2. Upload "final rough and rhyme hindi.jsonl" as a Kaggle Dataset.
3. Run Cell 1, RESTART KERNEL, then run Cell 2 onward.
4. After training, download Output → sarvam-hindi-lyrics/

Cell 1 – install dependencies  ← restart kernel after this
──────────────────────────────
!pip install -q --upgrade transformers accelerate peft bitsandbytes sentencepiece trl
"""

# ── Cell 2 – imports & config ────────────────────────────────────────────────
import json, random
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset

# ── EDIT THIS PATH to match your Kaggle dataset mount ────────────────────────
INPUT = '/kaggle/input/<your-dataset-name>/final rough and rhyme hindi.jsonl'
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = '/kaggle/working/sarvam-hindi-lyrics'
MODEL_NAME = 'sarvamai/sarvam-2b-v0.5'
MAX_SEQ_LEN = 512
SEED = 42

# Response marker — must be ASCII so tokenizer splits it cleanly
RESPONSE_TEMPLATE = '### Response:\n'

random.seed(SEED)
print(f'CUDA available : {torch.cuda.is_available()}')
print(f'GPUs           : {torch.cuda.device_count()}')

# ── Cell 3 – load tokenizer ──────────────────────────────────────────────────
print(f'Loading tokenizer from {MODEL_NAME}…')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'   # needed for causal LM training
print('Tokenizer loaded.')

# ── Cell 4 – load model in 4-bit (QLoRA) ────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = 'nf4',
    bnb_4bit_compute_dtype    = torch.float16,
    bnb_4bit_use_double_quant = True,
)

print(f'Loading {MODEL_NAME} in 4-bit…')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = bnb_cfg,
    device_map          = 'auto',
    torch_dtype         = torch.float16,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_cfg = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    bias           = 'none',
    task_type      = 'CAUSAL_LM',
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                      'gate_proj', 'up_proj', 'down_proj'],
)
model = get_peft_model(model, lora_cfg)
# Required for gradient checkpointing to work with PEFT/QLoRA
model.enable_input_require_grads()
model.print_trainable_parameters()
print('Model loaded.\n')

# ── Cell 5 – prompt builder ──────────────────────────────────────────────────
GENRE_STYLE = {
    'devotional': 'भक्ति और आस्था से भरी',
    'romantic'  : 'प्रेम और रोमांस से भरी',
    'sad'       : 'दुख और विरह से भरी',
    'happy'     : 'खुशी और उल्लास से भरी',
    'party'     : 'उत्साह और जोश से भरी',
}

def build_prompt(rec, include_output=True):
    genre      = rec.get('genre', 'hindi')
    style_desc = GENRE_STYLE.get(genre, genre)
    rhyme_sch  = rec.get('rhyme_scheme', '?')
    rhyme_grp  = rec.get('rhyme_group', '?')
    prev_end   = rec.get('prev_end', 'none')
    rough      = rec.get('rough_hindi', '').strip()
    output     = rec.get('output', '').strip()

    prev_end_line = (
        f'पिछली पंक्ति का अंत: "{prev_end}"'
        if prev_end and prev_end.lower() != 'none'
        else 'यह पहली पंक्ति है।'
    )

    prompt = (
        f'### Instruction:\n'
        f'नीचे दिए गए हिंदी अनुवाद को एक {style_desc} गाने की काव्यात्मक पंक्ति में बदलें।\n'
        f'Genre: {genre} | Rhyme scheme: {rhyme_sch} | Rhyme group: {rhyme_grp}\n'
        f'{prev_end_line}\n\n'
        f'### Input:\n{rough}\n\n'
        f'{RESPONSE_TEMPLATE}'
    )
    if include_output:
        prompt += output + tokenizer.eos_token
    return prompt

# ── Cell 6 – dataset ─────────────────────────────────────────────────────────
with open(INPUT, encoding='utf-8') as f:
    records = [json.loads(l) for l in f if l.strip()]

records = [r for r in records
           if r.get('rough_hindi', '').strip() and r.get('output', '').strip()]

random.shuffle(records)
split      = int(len(records) * 0.9)
train_recs = records[:split]
val_recs   = records[split:]
print(f'Train : {len(train_recs)}')
print(f'Val   : {len(val_recs)}')

class LyricsDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, idx):
        rec         = self.recs[idx]
        full_text   = build_prompt(rec, include_output=True)
        prompt_only = build_prompt(rec, include_output=False)

        full_ids   = tokenizer(full_text,   truncation=True, max_length=MAX_SEQ_LEN)['input_ids']
        prompt_len = len(tokenizer(prompt_only, truncation=True, max_length=MAX_SEQ_LEN)['input_ids'])

        # Mask all prompt tokens in the labels so loss is only on the lyric output
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            'input_ids':      torch.tensor(full_ids,          dtype=torch.long),
            'attention_mask': torch.ones(len(full_ids),        dtype=torch.long),
            'labels':         torch.tensor(labels,            dtype=torch.long),
        }

train_ds = LyricsDataset(train_recs)
val_ds   = LyricsDataset(val_recs)

# ── Cell 7 – data collator (pads inputs with pad_token_id, labels with -100) ─
def collate_fn(batch):
    input_ids      = pad_sequence([b['input_ids']      for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
    labels         = pad_sequence([b['labels']         for b in batch], batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# ── Cell 8 – training ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 8,    # effective batch = 32
    warmup_steps                = 200,
    learning_rate               = 2e-4,
    weight_decay                = 0.01,
    fp16                              = True,
    gradient_checkpointing            = True,
    gradient_checkpointing_kwargs     = {'use_reentrant': False},
    eval_strategy                     = 'epoch',
    save_strategy               = 'epoch',
    load_best_model_at_end      = True,
    metric_for_best_model       = 'eval_loss',
    greater_is_better           = False,
    logging_steps               = 50,
    save_total_limit            = 2,
    report_to                   = 'none',
    seed                        = SEED,
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_ds,
    eval_dataset  = val_ds,
    data_collator = collate_fn,
)

print('Starting training…')
trainer.train()

# ── Cell 9 – save ────────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f'\nFine-tuned model saved to {OUTPUT_DIR}')
print('Download the folder from the Output tab.')

# ── Cell 10 – sanity check ───────────────────────────────────────────────────
print('\n── Sanity check (5 val samples) ──')
model.eval()
device = next(model.parameters()).device

for rec in val_recs[:5]:
    prompt = build_prompt(rec, include_output=False)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=MAX_SEQ_LEN).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens    = 80,
            num_beams         = 4,
            no_repeat_ngram_size = 3,
            early_stopping    = True,
            pad_token_id      = tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    generated  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f'  GENRE  : {rec.get("genre")}  | rhyme_group: {rec.get("rhyme_group")}  | prev_end: {rec.get("prev_end")}')
    print(f'  ROUGH  : {rec["rough_hindi"].strip()[:100]}')
    print(f'  FINETUNED: {generated[:100]}')
    print(f'  TARGET : {rec["output"].strip()[:100]}')
    print()
