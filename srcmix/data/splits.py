from sklearn.model_selection import train_test_split
from datasets import Dataset

def create_splits(df, val_ratio=0.02, seed=42):
    train_df, temp_df = train_test_split(df, test_size=val_ratio, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
        Dataset.from_pandas(test_df.reset_index(drop=True))
    )