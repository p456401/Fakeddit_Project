import pandas as pd
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import time


DATA_PATH = "data/processed/train_clean.csv"
IMAGE_DIR = Path("data/images/train")
FAILED_FILE = Path("data/images/failed_downloads.csv")

MAX_IMAGES = 1000


IMAGE_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)

# Only process first MAX_IMAGES for the initial test
df = df.head(MAX_IMAGES)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

success = 0
failed = 0
skipped = 0

failed_records = []


for i, row in enumerate(df.itertuples(index=False), start=1):

    image_id = str(row.id)
    image_url = row.image_url

    output_path = IMAGE_DIR / f"{image_id}.jpg"

    # Skip if already downloaded
    if output_path.exists():
        skipped += 1
        continue

    try:

        response = session.get(
            image_url,
            timeout=10
        )

        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        image.save(
            output_path,
            "JPEG"
        )

        success += 1

    except Exception as e:

        failed += 1

        failed_records.append({
            "id": image_id,
            "image_url": image_url,
            "error": str(e)
        })

    # Progress
    if i % 100 == 0:

        print(
            f"Processed: {i}/{len(df)} | "
            f"Success: {success} | "
            f"Failed: {failed} | "
            f"Skipped: {skipped}"
        )


# Save failed URLs
if failed_records:

    pd.DataFrame(
        failed_records
    ).to_csv(
        FAILED_FILE,
        index=False
    )


print("\nDownload completed.")
print("Total processed:", len(df))
print("Successful:", success)
print("Failed:", failed)