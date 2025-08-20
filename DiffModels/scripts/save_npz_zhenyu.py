import os
import cv2
import torch
import numpy as np
import random
import argparse
from torchvision import transforms
from PIL import Image

def set_seed(seed: int = 20) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_images_with_aug(directory, outpath, num, same_num, aug):
    files = sorted(os.listdir(directory))

    if aug:
        print("aug is enabled")
        trans = transforms.RandomPerspective(distortion_scale=0.2, p=1.0, fill=255, interpolation=Image.BILINEAR)

    count = 0
    for i, filename in enumerate(files):
        for j in range(same_num):

            image_path = os.path.join(directory, filename)
            image = Image.open(image_path).convert("L")
            
            if aug:
                image = trans(image)

            image = image.resize((128, 128))
            image.save(os.path.join(outpath, f'{count}.png'))
            
            count += 1
            if count >= num:
                break
        if count >= num:
                break



if __name__ == '__main__':

    set_seed(20)
    
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument(
        '--input', 
        type=str, 
        default="/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/test-images/image")
    parser.add_argument(
        '--outdir', 
        type=str,
        default="./output/test-large/label")
    parser.add_argument('--num', type=int, default=20)
    parser.add_argument('--same_num', type=int, default=20)
    parser.add_argument('--aug', action='store_true', default=False)
    args = parser.parse_args()

    # Example usage
    input_directory = args.input  # Replace with the path to your input directory
    save_directory = args.outdir

    os.makedirs(save_directory, exist_ok=True)

    # Load images from the directory
    assert args.num % args.same_num == 0, "num must be divisible by same_num"
    load_images_with_aug(input_directory, args.outdir, args.num, args.same_num, args.aug)

    print(f'Presudo ground truth images saved to {save_directory}.')

    
