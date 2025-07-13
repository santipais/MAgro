import os
import shutil
import random
import argparse

def split_dataset(image_dir, label_dir, output_dir, val_ratio=0.15, test_ratio=0.2, seed=42):
    random.seed(seed)

    valid_exts = ('.png', '.jpg', '.jpeg')

    # Create maps: base name → extension
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
    label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(valid_exts)]

    image_map = {os.path.splitext(f)[0]: os.path.splitext(f)[1] for f in image_files}
    label_map = {os.path.splitext(f)[0]: os.path.splitext(f)[1] for f in label_files}

    # Keep only base names that exist in both image and label directories
    common_names = sorted(list(set(image_map.keys()) & set(label_map.keys())))

    # Shuffle and split
    random.shuffle(common_names)

    total = len(common_names)
    test_size = int(total * test_ratio)
    trainval_names = common_names[test_size:]
    val_size = int(len(trainval_names) * val_ratio)

    test_names = set(common_names[:test_size])
    val_names = set(trainval_names[:val_size])
    train_names = set(trainval_names[val_size:])

    def copy_files(names, split):
        for name in names:
            img_ext = image_map[name]
            lbl_ext = label_map[name]

            img_src = os.path.join(image_dir, f'{name}{img_ext}')
            lbl_src = os.path.join(label_dir, f'{name}{lbl_ext}')

            img_dst = os.path.join(output_dir, 'images', split, f'{name}{img_ext}')
            lbl_dst = os.path.join(output_dir, 'annotations', split, f'{name}{lbl_ext}')

            os.makedirs(os.path.dirname(img_dst), exist_ok=True)
            os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)

            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)

    copy_files(train_names, 'train')
    copy_files(val_names, 'val')
    copy_files(test_names, 'test')

    print(f"Done. Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test.")
    parser.add_argument('--val', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    if args.val + args.test >= 1.0:
        raise ValueError(f"val + test must be < 1.0 (currently {args.val + args.test})")

    # 🔒 Fixed paths (relative to current working directory)
    cwd = os.getcwd()
    image_dir = os.path.join(cwd, 'data', 'images')
    label_dir = os.path.join(cwd, 'data', 'labels')
    output_dir = os.path.join(cwd, 'data', 'malezas')

    split_dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=output_dir,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
