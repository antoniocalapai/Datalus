import os
import random
import shutil

BASE = "/Users/acalapai/Desktop/ElmJok"
RAW_ROOT = os.path.join(BASE, "RAW")
PREP_ROOT = os.path.join(BASE, "PREP")

MONKEYS = ["Elm", "Jok"]
SPLIT = 0.8  # 80% train, 20% test

for m in MONKEYS:
    raw_dir = os.path.join(RAW_ROOT, m)

    # NEW correct structure
    train_dir = os.path.join(PREP_ROOT, "train", m)
    test_dir  = os.path.join(PREP_ROOT, "test", m)

    # Create folders if missing
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List images
    images = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not images:
        print(f"[WARNING] No images found for {m} in {raw_dir}")
        continue

    random.shuffle(images)
    split_idx = int(len(images) * SPLIT)

    train_imgs = images[:split_idx]
    test_imgs  = images[split_idx:]

    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(raw_dir, img),
                    os.path.join(train_dir, img))

    for img in test_imgs:
        shutil.copy(os.path.join(raw_dir, img),
                    os.path.join(test_dir, img))

    print(f"{m}: {len(train_imgs)} train, {len(test_imgs)} test images")

    import os

    BASE = "/Users/acalapai/Desktop/ElmJok"
    RAW_ROOT = os.path.join(BASE, "RAW")
    PREP_ROOT = os.path.join(BASE, "PREP")

    MONKEYS = ["Elm", "Jok"]
    EXT = (".png", ".jpg", ".jpeg")


    def count_images(path):
        if not os.path.exists(path):
            return 0
        return len([f for f in os.listdir(path) if f.lower().endswith(EXT)])


    print("=== VERIFICATION OF TRAIN/TEST SPLIT ===\n")

    for m in MONKEYS:
        raw_dir = os.path.join(RAW_ROOT, m)

        # NEW correct structure
        train_dir = os.path.join(PREP_ROOT, "train", m)
        test_dir = os.path.join(PREP_ROOT, "test", m)

        raw_count = count_images(raw_dir)
        train_count = count_images(train_dir)
        test_count = count_images(test_dir)

        if raw_count == 0:
            print(f"{m}: No RAW images found — cannot verify.\n")
            continue

        train_pct = (train_count / raw_count) * 100
        test_pct = (test_count / raw_count) * 100

        print(f"Monkey: {m}")
        print(f"  RAW images   : {raw_count}")
        print(f"  Train images : {train_count}  ({train_pct:.2f}%)")
        print(f"  Test images  : {test_count}   ({test_pct:.2f}%)")

        if abs(train_pct - 80) < 1 and abs(test_pct - 20) < 1:
            print("Split is correct (≈80/20)\n")
        else:
            print("Split deviates from 80/20\n")