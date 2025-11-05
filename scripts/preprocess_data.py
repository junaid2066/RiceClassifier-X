import os, random, shutil

def split_data(raw_dir='data/raw', processed_dir='data/processed', train_ratio=0.7, val_ratio=0.15):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(processed_dir, split), exist_ok=True)

    for cls in os.listdir(raw_dir):
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        images = os.listdir(cls_dir)
        random.shuffle(images)

        train_split = int(len(images)*train_ratio)
        val_split = int(len(images)*(train_ratio+val_ratio))

        splits = {'train': images[:train_split], 'val': images[train_split:val_split], 'test': images[val_split:]}

        for split, files in splits.items():
            split_cls_dir = os.path.join(processed_dir, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(cls_dir, f), os.path.join(split_cls_dir, f))

if __name__ == '__main__':
    split_data()
    print('✅ Data split completed successfully!')
