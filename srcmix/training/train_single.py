import argparse
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from srcmix.data.build_dataset import load_excel, build_single_language_dataset
from srcmix.data.splits import train_dev_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--sheet_name", required=True)
    parser.add_argument("--source_language", required=True)
    parser.add_argument("--target_language", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_excel(args.data_file, args.sheet_name)
    dataset = build_single_language_dataset(df, args.source_language, args.target_language)

    train_dataset, _ = train_dev_split(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess(example):
        inputs = [f"Translate {args.source_language} to {args.target_language}: {text}"
                  for text in example[args.source_language]]
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=256)

        labels = tokenizer(text_target=example[args.target_language],
                           truncation=True,
                           padding="max_length",
                           max_length=256)

        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora_config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=1e-3,
        num_train_epochs=5,
        save_strategy="no",
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":
    main()