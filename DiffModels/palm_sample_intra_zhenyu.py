"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
import cv2
import inspect
import random
import torch
import blobfile as bf
import numpy as np
from natsort import natsorted
import torch as th
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

class ImageDataset(Dataset):
    """
    Load low-resolution palm images from a directory *lazily* so that we do not
    fill system RAM with the whole .npz array.  All files whose suffix is
    png/jpg/jpeg/bmp will be included.

    If `class_cond=True`, the label is parsed from the file name: everything
    before the first '_' is treated as an integer category; otherwise a scalar
    0 will be returned.
    """
    def __init__(self, root_dir: str, class_cond: bool = False):
        self.root_dir: str = root_dir
        self.class_cond: bool = class_cond
        exts: List[str] = [".png", ".jpg", ".jpeg", ".bmp"]
        self.paths: List[str] = [
            os.path.join(root_dir, fn)
            for fn in os.listdir(root_dir)
            if os.path.splitext(fn)[1].lower() in exts
        ]
        if len(self.paths) == 0:
            raise RuntimeError(f"No image files found under {root_dir}")
        self.paths = natsorted(self.paths) # fixed order for reproducibility


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # BGR, H×W×3, uint8
        if img is None:
            raise RuntimeError(f"cv2 could not read {path}")
        img = img.astype(np.float32) / 127.5 - 1.0        # → [-1,1]
        img = img[..., None]
        img = torch.from_numpy(img).permute(2, 0, 1)      # C,H,W
        sample = {"low_res": img}
        if self.class_cond:
            class_id = int(os.path.basename(path).split("_")[0])
            sample["y"] = torch.tensor(class_id, dtype=torch.long)
        return sample, idx


def set_seed(seed: int = 20) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_with_noise_sharing(
        diffusion, 
        model, 
        shape, 
        sharing_params,
        noise=None, 
        clip_denoised=True, 
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        last_K=True):

    final = None
    count = 0
    for sample in diffusion.p_sample_loop_progressive_with_noise_sharing(
        model,
        shape,
        sharing_params=sharing_params,
        noise=noise,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        cond_fn=cond_fn,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
        last_K=last_K
    ):
        final = sample
        count += 1
    return final["sample"]


def main():
    set_seed(20)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, palm_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log(f"{dist_util.dev()} is used.")

    logger.log("loading data...")
    data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)

    logger.log("creating samples...")
    assert args.batch_size % args.sharing_num == 0
    count = 0
    os.makedirs(args.outdir, exist_ok=True)
    while count < args.num_samples:
        model_kwargs, indices = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        sharing_params = (args.sharing_num, args.sharing_step)

        sample = sample_with_noise_sharing(
            diffusion=diffusion,
            model=model,
            shape=(args.batch_size, 3, args.large_size, args.large_size),
            sharing_params=sharing_params,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1) # b, h, w, c
        sample = sample.contiguous()

        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        indices = indices.to(dist_util.dev(), non_blocking=True).long()
        all_idx = [th.zeros_like(indices) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL
        dist.all_gather(all_idx, indices)

        # only rank‑0 is responsible for writing to disk, to avoid duplicates
        if dist.get_rank() == 0:
            flat_samples = th.cat(all_samples, dim=0)  # (world*B , H , W , C)
            flat_idx = th.cat(all_idx)
            order = th.argsort(flat_idx)
            ordered_imgs = flat_samples[order]
            ordered_idx = flat_idx[order]

            for gid, img_tensor in zip(ordered_idx.tolist(), ordered_imgs):
                img = img_tensor.cpu().numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.outdir, f"{gid}.png"), img)

        if dist.get_rank() == 0:
            logger.log(f"created {ordered_idx[-1] + 1} samples")

        count += args.batch_size * dist.get_world_size()

    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(img_dir, batch_size, class_cond):
    """
    Build a PyTorch Dataset + DistributedSampler so that images are streamed
    from disk on demand instead of loading a huge npz into RAM.  This function
    returns an **infinite generator** that yields dictionaries compatible with
    the original training / sampling code.
    """
    dataset = ImageDataset(img_dir, class_cond=class_cond)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    while True:
        sampler.set_epoch(random.randint(0, 10_000_000))
        for batch in loader:
            yield batch


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=128,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    res.update(diffusion_defaults())
    return res


def palm_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 128
    res["small_size"] = 128
    res["in_channels"] = 6
    res["out_channels"] = 3

    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=20,
        use_ddim=False,
        base_samples="./output/test-large/data.npz",
        model_path="./checkpoint/diffusion-netpalm-scale-128/ema_0.9999.pt",
        outdir = "./output/test-large/results",
        sharing_num=20,
        sharing_step=500,
    )
    defaults.update(palm_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
