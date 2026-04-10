"""
finetune_indicbart.py  ─  Kaggle edition
─────────────────────────────────────────
Fine-tunes ai4bharat/IndicBART on rough_hindi → output pairs so the model
learns to convert flat Hindi translations into polished rhyming Hindi lyrics.

HOW TO USE ON KAGGLE
────────────────────
1. Create a new Kaggle notebook (GPU T4 x2, internet ON).
2. Upload "final rough and rhyme hindi.jsonl" as a Kaggle Dataset.
3. Run Cell 1, RESTART KERNEL, then run Cell 2 onward.
4. After training, download the saved model from Output → indicbart-hindi-lyrics/

Cell 1 – install dependencies  ← restart kernel after this
──────────────────────────────
!pip install -q --upgrade transformers accelerate sentencepiece sacremoses peft
"""

# ── Cell 2 – imports & config ────────────────────────────────────────────────
import json, random
import torch
from torch.utils.data import Dataset
from transformers import (
    MBartForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# ── EDIT THIS PATH to match your Kaggle dataset mount ────────────────────────
INPUT      = '/kaggle/input/<your-dataset-name>/final rough and rhyme hindi.jsonl'
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = '/kaggle/working/indicbart-hindi-lyrics'
MODEL_NAME  = 'ai4bharat/IndicBART'
HINDI_TAG   = '<2hi>'
MAX_SRC_LEN = 256
MAX_TGT_LEN = 256
SEED        = 42

random.seed(SEED)
print(f'CUDA available : {torch.cuda.is_available()}')
print(f'GPUs           : {torch.cuda.device_count()}')

# ── Cell 3 – load tokenizer & model ─────────────────────────────────────────
print(f'Loading {MODEL_NAME}…')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model     = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

# Tell the model to always start decoding with the Hindi language token
HINDI_BOS_ID = tokenizer.convert_tokens_to_ids(HINDI_TAG)
model.config.decoder_start_token_id          = HINDI_BOS_ID
model.generation_config.forced_bos_token_id  = HINDI_BOS_ID
model.generation_config.decoder_start_token_id = HINDI_BOS_ID
print(f'Hindi BOS token id : {HINDI_BOS_ID}')
print('Model loaded.\n')

# ── Cell 4 – dataset ─────────────────────────────────────────────────────────
class LyricsDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        src = HINDI_TAG + ' ' + rec['rough_hindi'].strip()
        tgt = rec['output'].strip()

        # Tokenize source
        enc = tokenizer(
            src,
            max_length=MAX_SRC_LEN,
            truncation=True,
        )
        enc.pop('token_type_ids', None)

        # Tokenize target (labels)
        dec = tokenizer(
            tgt,
            max_length=MAX_TGT_LEN,
            truncation=True,
        )

        enc['labels'] = dec['input_ids']
        return enc

# ── Cell 5 – load data & train/val split ─────────────────────────────────────
with open(INPUT, encoding='utf-8') as f:
    records = [json.loads(l) for l in f if l.strip()]

records = [r for r in records
           if r.get('rough_hindi', '').strip() and r.get('output', '').strip()]

random.shuffle(records)
split       = int(len(records) * 0.9)
train_recs  = records[:split]
val_recs    = records[split:]

train_ds = LyricsDataset(train_recs)
val_ds   = LyricsDataset(val_recs)
print(f'Train : {len(train_ds)}')
print(f'Val   : {len(val_ds)}')

# ── Cell 6 – training arguments ──────────────────────────────────────────────
collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,   # ignore padding positions in loss
)

training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 10,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 4,    # effective batch size = 64
    warmup_steps                = 300,
    weight_decay                = 0.01,
    fp16                        = True,
    gradient_checkpointing      = True,
    predict_with_generate       = True,
    generation_max_length       = MAX_TGT_LEN,
    generation_num_beams        = 4,
    eval_strategy               = 'epoch',
    save_strategy               = 'epoch',
    load_best_model_at_end      = True,
    metric_for_best_model       = 'eval_loss',
    greater_is_better           = False,
    logging_steps               = 100,
    save_total_limit            = 2,
    dataloader_num_workers      = 2,
    dataloader_pin_memory       = False,
    report_to                   = 'none',
    seed                        = SEED,
)

trainer = Seq2SeqTrainer(
    model             = model,
    args              = training_args,
    train_dataset     = train_ds,
    eval_dataset      = val_ds,
    processing_class  = tokenizer,
    data_collator     = collator,
)

# ── Cell 7 – train ───────────────────────────────────────────────────────────
print('Starting training…')
trainer.train()

# ── Cell 8 – save ────────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f'\nFine-tuned model saved to {OUTPUT_DIR}')
print('Download the folder from the Output tab.')

# ── Cell 9 – quick sanity check ──────────────────────────────────────────────
print('\n── Sanity check (5 val samples) ──')
model.eval()
device = next(model.parameters()).device

for rec in val_recs[:5]:
    src    = HINDI_TAG + ' ' + rec['rough_hindi'].strip()
    inputs = tokenizer(src, return_tensors='pt',
                       truncation=True, max_length=MAX_SRC_LEN).to(device)
    inputs.pop('token_type_ids', None)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=HINDI_BOS_ID,
            num_beams=4,
            max_new_tokens=MAX_TGT_LEN,
            no_repeat_ngram_size=3,
        )

    generated = tokenizer.decode(out[0], skip_special_tokens=True).replace(HINDI_TAG, '').strip()
    print(f'  ROUGH  : {rec["rough_hindi"].strip()[:100]}')
    print(f'  FINETUNED: {generated[:100]}')
    print(f'  TARGET : {rec["output"].strip()[:100]}')
    print()
