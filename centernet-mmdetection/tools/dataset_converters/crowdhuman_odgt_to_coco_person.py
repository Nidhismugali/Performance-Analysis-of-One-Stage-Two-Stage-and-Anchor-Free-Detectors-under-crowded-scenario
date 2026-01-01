import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CrowdHuman .odgt to COCO (person full body only).')
    parser.add_argument('--odgt', required=True,
                        help='Path to annotation_train.odgt or annotation_val.odgt')
    parser.add_argument('--img-dir', required=True,
                        help='Root directory containing images (can have subfolders)')
    parser.add_argument('--out', required=True,
                        help='Output COCO json path')
    return parser.parse_args()

def main():
    args = parse_args()
    odgt_path = Path(args.odgt)
    img_root = Path(args.img_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = [{"id": 1, "name": "person"}]

    ann_id = 1
    img_id = 1

    with odgt_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # CrowdHuman ID looks like: "273271,1017c000ac1360b7"
            file_stem = rec.get('ID')
            file_name = file_stem
            if not file_name.lower().endswith('.jpg'):
                file_name = f'{file_name}.jpg'

            # Search recursively under img_root (train or val)
            img_path = None
            for candidate in img_root.rglob(file_name):
                img_path = candidate
                break

            if img_path is None:
                print(f'Warning: image not found for {file_name}, skipping')
                continue

            image_info = {
                'id': img_id,
                'file_name': str(img_path.relative_to(img_root)).replace('\\', '/'),
                'width': 0,
                'height': 0,
            }
            images.append(image_info)

            for box in rec.get('gtboxes', []):
                if box.get('tag') != 'person':
                    continue

                extra = box.get('extra', {})
                if extra.get('ignore', 0) == 1:
                    continue

                fbox = box.get('fbox', None)
                if fbox is None:
                    continue

                x, y, w, h = fbox
                if w <= 0 or h <= 0:
                    continue

                ann = {
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': 1,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'area': float(w * h),
                    'iscrowd': 0,
                }
                annotations.append(ann)
                ann_id += 1

            img_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    with out_path.open('w', encoding='utf-8') as f:
        json.dump(coco, f)
    print(f'Saved COCO annotations to {out_path}')
    print(f'Images: {len(images)}, Annotations: {len(annotations)}')

if __name__ == '__main__':
    main()
