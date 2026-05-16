import os
import sys
os.chdir(sys.path[0])
import argparse
from pathlib import Path
import torch
import pyiqa

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
DATASET="MSRS"
METHOD="S2Fusion"

def is_image_file(p):
    return Path(p).suffix.lower() in IMG_EXTS

def collect_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.iterdir() if p.is_file() and is_image_file(p)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=f"/home/DeepLearing/Imagefusion/FusionImageEvalution/Quantitative/Results/{DATASET}/{METHOD}", help='单张图像路径或文件夹路径')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric_pi = pyiqa.create_metric('pi', device=device)

    image_list = collect_images(args.input)
    if len(image_list) == 0:
        print('没有找到图像')
        return

    total = 0.0
    for img_path in image_list:
        score = float(metric_pi(str(img_path)).cpu().item())
        total += score
        print(f'{img_path.name}: PI = {score:.6f}')

    print(f'\nAverage PI = {total / len(image_list):.6f}')

if __name__ == '__main__':
    main()