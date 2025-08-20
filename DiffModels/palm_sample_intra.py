"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
import inspect
import random
import torch
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

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
    all_images = []
    assert args.batch_size % args.sharing_num == 0
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
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
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


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
        sharing_num=20,
        sharing_step=500,
    )
    defaults.update(palm_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
