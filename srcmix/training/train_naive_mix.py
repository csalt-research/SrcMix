import argparse
import random
import numpy as np
import pandas as pd
import torch

from itertools import permutations
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


# -------------------------
# Argument Parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--sheet_name", required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_adapter_path", required=True)

    return parser.parse_args()


# -------------------------
# Seed Setup
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Build Naïve M2M Dataset
# -------------------------
def build_m2m_dataset(df, languages):
    pairs = []

    for src, tgt in permutations(languages, 2):
        if src in df.columns and tgt in df.columns:
            sub = df[[src, tgt]].dropna().copy()
            sub.rename(columns={src: "source", tgt: "target"}, inplace=True)
            sub["src_lang"] = src
            sub["tgt_lang"] = tgt
            pairs.append(sub)

    if len(pairs) == 0:
        raise ValueError("No valid language pairs found.")

    df_all = pd.concat(pairs, ignore_index=True)
    return Dataset.from_pandas(df_all)


# -------------------------
# Preprocessing
# -------------------------
def preprocess_fn(batch, tokenizer, max_src_len, max_tgt_len):
    prompts = [
        f"Translate {sl} to {tl}: {s}"
        for sl, tl, s in zip(batch["src_lang"], batch["tgt_lang"], batch["source"])
    ]

    model_inputs = tokenizer(
        prompts,
        max_length=max_src_len,
        padding="max_length",
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["target"],
        max_length=max_tgt_len,
        padding="max_length",
        truncation=True,
    )

    labels["input_ids"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in lab]
        for lab in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    set_seed()

    df = pd.read_excel(args.data_file, sheet_name=args.sheet_name)

    dataset = build_m2m_dataset(df, args.languages)

    train_df, val_df = train_test_split(
        dataset.to_pandas(), test_size=0.02, random_state=42
    )

    train_hf = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_hf = Dataset.from_pandas(val_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Estimate lengths
    sample_src = train_hf["source"][:500]
    sample_tgt = train_hf["target"][:500]

    max_src_len = int(np.percentile(
        [len(tokenizer(s)["input_ids"]) for s in sample_src], 85
    )) if sample_src else 256

    max_tgt_len = int(np.percentile(
        [len(tokenizer(s)["input_ids"]) for s in sample_tgt], 90
    )) if sample_tgt else 128

    tok_train = train_hf.map(
        lambda x: preprocess_fn(x, tokenizer, max_src_len, max_tgt_len),
        batched=True,
        remove_columns=train_hf.column_names,
    )

    tok_val = val_hf.map(
        lambda x: preprocess_fn(x, tokenizer, max_src_len, max_tgt_len),
        batched=True,
        remove_columns=val_hf.column_names,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3,
        num_train_epochs=3,
        logging_steps=500,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
        train_dataset=tok_train,
        eval_dataset=tok_val,
    )

    trainer.train()

    model.save_pretrained(args.save_adapter_path)
    tokenizer.save_pretrained(args.save_adapter_path)


if __name__ == "__main__":
    main()