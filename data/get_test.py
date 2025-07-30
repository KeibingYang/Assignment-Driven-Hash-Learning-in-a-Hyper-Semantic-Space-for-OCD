import os
import json
from pathlib import Path

ROOT_PUBLIC_TEST = "/seu_nvme/home/fenglei/213240634/SMILE_20250604002666/SMILE/data/dataset/inat2021_data/public_test"
JSON_PUBLIC_TEST = "/seu_nvme/home/fenglei/213240634/SMILE_20250604002666/SMILE/data/dataset/inat2021_data/public_test.json"
OUT_SYMLINK_DIR  = "/seu_nvme/home/fenglei/213240634/SMILE_20250604002666/SMILE/data/subsets_public_test"

SUPER_CATEGORIES = ['Mollusks', 'Arachnids', 'Animalia']

def main():
    with open(JSON_PUBLIC_TEST, "r") as f:
        anno = json.load(f)
    image_id_to_file = {img['id']: img['file_name'] for img in anno['images']}
    # 获取每个类别id的supercategory
    cat_id_to_supercat = {cat['id']: cat['supercategory'] for cat in anno['categories']}
    
    # 将图片分配到三大类
    class_file_dict = {c: [] for c in SUPER_CATEGORIES}
    for ann in anno['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        supercat = cat_id_to_supercat.get(cat_id)
        if supercat in SUPER_CATEGORIES:
            file_name = image_id_to_file[img_id]
            class_file_dict[supercat].append(file_name)
    
    # 建符号链接
    for supercat, files in class_file_dict.items():
        out_dir = Path(OUT_SYMLINK_DIR) / supercat
        out_dir.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            src = Path(ROOT_PUBLIC_TEST) / file_name
            link = out_dir / file_name
            try:
                if not link.exists():
                    os.symlink(src, link)
            except Exception as e:
                print(f"失败: {src} -> {link}: {e}")

if __name__ == '__main__':
    main()
