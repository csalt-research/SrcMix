import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from srcmix.data.build_dataset import load_excel
from srcmix.data.mixing import build_srcmix_dataset
from srcmix.data.splits import train_dev_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--sheet_name", required=True)
    parser.add_argument("--source_languages", nargs="+", required=True)
    parser.add_argument("--target_language", required=True)
    parser.add_argument("--mixing_ratio", type=float, default=1.0)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_excel(args.data_file, args.sheet_name)
    dataset = build_srcmix_dataset(
        df,
        args.source_languages,
        args.target_language,
        args.mixing_ratio
    )

    train_dataset, _ = train_dev_split(dataset)

    print("SrcMix dataset size:", len(train_dataset))
    print("Training logic identical to train_single.py")

if __name__ == "__main__":
    main()