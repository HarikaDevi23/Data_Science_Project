import os
import shutil
import random

SRC_DIR = "Dataset"
DST_DIR = "DatasetSampled"
SPLITS = ["Train", "Validation", "Test"]
SAMPLES_PER_CLASS = 1000

random.seed(42)

for split in SPLITS:
    src_split = os.path.join(SRC_DIR, split)
    dst_split = os.path.join(DST_DIR, split)
    if not os.path.exists(src_split):
        print(f"Source split not found: {src_split}")
        continue
    classes = [d for d in os.listdir(src_split) if os.path.isdir(os.path.join(src_split, d))]
    for cls in classes:
        src_cls = os.path.join(src_split, cls)
        dst_cls = os.path.join(dst_split, cls)
        os.makedirs(dst_cls, exist_ok=True)
        images = [f for f in os.listdir(src_cls) if os.path.isfile(os.path.join(src_cls, f))]
        sample = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))
        print(f"Copying {len(sample)} images for {split}/{cls}...")
        for img in sample:
            shutil.copy2(os.path.join(src_cls, img), os.path.join(dst_cls, img))
print("\nSampling complete! Sampled dataset is in DatasetSampled/") 