import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_dataset(input_file, output_file):
    print(f"\nProcessing: {input_file}")

    df = pd.read_csv(RAW_DIR / input_file, sep="\t")

    original_rows = len(df)

    # Remove samples without an image URL
    df = df.dropna(subset=["image_url"])

    after_image_filter = len(df)

    # Remove duplicate IDs
    df = df.drop_duplicates(subset=["id"])

    after_duplicates = len(df)

    # Keep only required columns
    df = df[
        [
            "id",
            "clean_title",
            "image_url",
            "2_way_label",
            "subreddit",
            "domain",
        ]
    ].copy()

    # Rename columns
    df = df.rename(
        columns={
            "clean_title": "text",
            "2_way_label": "label",
        }
    )

    # Save processed dataset
    df.to_csv(PROCESSED_DIR / output_file, index=False)

    print(f"Original rows: {original_rows}")
    print(f"After removing missing images: {after_image_filter}")
    print(f"After removing duplicates: {after_duplicates}")
    print(f"Saved rows: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":

    clean_dataset(
        "multimodal_train.tsv",
        "train_clean.csv",
    )

    clean_dataset(
        "multimodal_validate.tsv",
        "validation_clean.csv",
    )

    clean_dataset(
        "multimodal_test_public.tsv",
        "test_clean.csv",
    )

    print("\nAll datasets processed successfully!")