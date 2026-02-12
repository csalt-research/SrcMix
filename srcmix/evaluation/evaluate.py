import argparse
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Argument Parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_language", required=True)
    parser.add_argument("--target_language", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--sheet_name", required=True)
    parser.add_argument("--peft_model_path", required=True)
    parser.add_argument("--output_file", required=True)

    return parser.parse_args()


# -------------------------
# Build Dataset
# -------------------------
def build_dataset(df, source_lang, target_lang):
    df = df[[source_lang, target_lang]].dropna().copy()
    df.rename(columns={source_lang: "source", target_lang: "target"}, inplace=True)
    return Dataset.from_pandas(df.reset_index(drop=True))


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    set_seed()

    print("Loading test data...")
    df = pd.read_excel(args.data_file, sheet_name=args.sheet_name)

    dataset = build_dataset(df, args.source_language, args.target_language)

    print("Loading model...")

    config = PeftConfig.from_pretrained(args.peft_model_path)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model.eval()

    bleu = BLEU()
    chrf = CHRF(word_order=2, beta=2)

    predictions = []
    references = []

    print("Generating translations...")

    for row in tqdm(dataset):
        prompt = f"Translate {args.source_language} to {args.target_language}: {row['source']}"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=100,
                num_beams=5
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(pred.strip())
        references.append([row["target"].strip()])

    # Save predictions
    result_df = pd.DataFrame({
        "Reference": [r[0] for r in references],
        "Prediction": predictions
    })
    result_df.to_excel(args.output_file, index=False)

    print("\nEvaluation Results")
    print("-" * 40)
    print("BLEU:", bleu.corpus_score(predictions, references))
    print("ChrF++:", chrf.corpus_score(predictions, references))

    with open(args.output_file.replace(".xlsx", "_scores.txt"), "w") as f:
        f.write(f"BLEU: {bleu.corpus_score(predictions, references)}\n")
        f.write(f"ChrF++: {chrf.corpus_score(predictions, references)}\n")

    print("Results saved.")


if __name__ == "__main__":
    main()