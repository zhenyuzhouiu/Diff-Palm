import numpy as np
import torch
import multiprocessing
import random
from polypalm import PolyPalmCreator
from copy import deepcopy
import os, shutil
import argparse
import time

def task_function(idx: int, imgnamelist: list, npznamelist: list, figcreator: PolyPalmCreator):
    figcreator.set_seed(idx)
    for i, f in enumerate(imgnamelist):
        figcreator.draw_by_multi(imgnamelist[i], npznamelist[i], scale=1.0)

        if i % 10 == 0:
            print(f"{idx} --> {f} saved!")


if __name__ == "__main__":

    start_time = time.time()  # 记录开始时间

    parser = argparse.ArgumentParser(description="Compare results")
    parser.add_argument("--ids", type=int, default=2)  # 2,000,000
    parser.add_argument("--output", type=str, default="test-images")
    parser.add_argument("--nproc", type=int, default=16)
    args = parser.parse_args()

    total_ids = args.ids
    start = 0
    num_processes = args.nproc

    step = total_ids // num_processes

    imsize = 256
    noise_weight = 0.3
    draw_thickness = random.randint(20, 25)
    is_prolong = False
    curve_1_is_cut = False
    curve_2_is_cut = False
    fig = PolyPalmCreator(
        imsize=imsize, noise_weight=noise_weight, draw_thickness=draw_thickness
    )

    results_dir = args.output
    os.makedirs(results_dir, exist_ok=True)
    shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    image_dir = os.path.join(results_dir, "image")
    os.makedirs(image_dir)
    params_dir = os.path.join(results_dir, "parameters")
    os.makedirs(params_dir)

    fig.fit("./labeled_data.pkl", args.output)
    

    tasks = []
    for i in range(num_processes):
        imgnamelist, npznamelist = [], []
        if i != num_processes - 1:
            for j in range(start + i * step, start + (i + 1) * step):
                imgnamelist.append(os.path.join(image_dir, f"{j}_0.png"))
                npznamelist.append(os.path.join(params_dir, f"{j}_0.npz"))
        else:
            for j in range(start + i * step, start + total_ids):
                imgnamelist.append(os.path.join(image_dir, f"{j}_0.png"))
                npznamelist.append(os.path.join(params_dir, f"{j}_0.npz"))

        tasks.append((i, deepcopy(imgnamelist), deepcopy(npznamelist), deepcopy(fig)))

    pool = multiprocessing.Pool(processes=num_processes)

    pool.starmap(task_function, tasks)

    pool.close()
    pool.join()

    print("total {} images".format(len(os.listdir(results_dir))))
    end_time = time.time()
    execution_time = end_time - start_time
    print("Total execution time:", execution_time, "seconds")
