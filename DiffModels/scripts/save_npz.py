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
    images = []
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

            image.save(os.path.join(outpath, f'{count}.png'))

            image = image.resize((128, 128))
            image = np.float32(np.array(image))
            images.append(image[None, :, :, None])
            
            count += 1
            if len(images) >= num:
                break
        if len(images) >= num:
                break


    return images[:num]


if __name__ == '__main__':

    set_seed(20)
    
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument(
        '--input', 
        type=str, 
        default="/home/ra1/Project/ZZY/Diff-Palm/PolyCreases/test-images/image")
    parser.add_argument(
        '--outnpz', 
        type=str,
        default="./output/test-large/data.npz")
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
    output_file = args.outnpz  # Replace with the desired path and name for the output npz file
    save_directory = args.outdir

    os.makedirs(save_directory, exist_ok=True)

    # Load images from the directory
    assert args.num % args.same_num == 0, "num must be divisible by same_num"
    images = load_images_with_aug(input_directory, args.outdir, args.num, args.same_num, args.aug)

    # Concatenate the images
    concatenated_images = np.concatenate(images, axis=0)

    print(concatenated_images.shape)

    # Save the concatenated images to a npz file
    if os.path.exists(output_file):
        os.remove(output_file)    
    np.savez(output_file, concatenated_images)
    print(f'Concatenated images saved to {output_file}.')

    
