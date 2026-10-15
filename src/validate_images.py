from pathlib import Path
from PIL import Image

IMAGE_DIR = Path("data/images/test_batch")

good = 0
bad = 0

for image_file in IMAGE_DIR.glob("*.jpg"):
    try:
        with Image.open(image_file) as img:
            img.verify()
        good += 1
    except Exception:
        bad += 1
        print("BAD:", image_file.name)

print("\nImage validation completed.")
print("Total files:", good + bad)
print("Valid images:", good)
print("Invalid images:", bad)