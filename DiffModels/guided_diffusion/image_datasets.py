import math
import os.path
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

import pywt
import cv2 as cv 


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def load_palm_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(os.path.join(data_dir, 'palm'))
    all_labels = _list_image_files_recursively(os.path.join(data_dir, 'label'))

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = PalmImageDataset(
        image_size,
        all_files,
        all_labels,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class PalmImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        label_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_labels = label_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        label_path = self.local_labels[idx]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        with bf.BlobFile(label_path, "rb") as f:
            pil_label = Image.open(f)
            pil_label.load()

        pil_image = pil_image.convert("RGB")
        pil_label = pil_label.convert("L")

        size = int(self.resolution * 1.1)
        #pil_image = pil_image.resize((size, size), Image.BILINEAR)
        #pil_label = pil_label.resize((size, size), Image.BILINEAR)
        pil_image = pil_image.resize((size, size), Image.BICUBIC)
        pil_label = pil_label.resize((size, size), Image.BICUBIC)

        if self.random_crop:
            crop_y = random.randrange(pil_image.size[0] - self.resolution + 1)
            crop_x = random.randrange(pil_image.size[0] - self.resolution + 1)
            arr = random_crop_arr_with_point(pil_image, self.resolution, crop_x, crop_y)
            arr_label = random_crop_arr_with_point(pil_label, self.resolution, crop_x, crop_y)
        else:
            arr = center_crop_arr(pil_image, self.resolution)
            arr_label = center_crop_arr(pil_label, self.resolution)

        if self.random_flip:
            if random.random() < 0.5:
                arr = arr[:, ::-1]
                arr_label = arr_label[:, ::-1]
        
        arr = arr.astype(np.float32) / 127.5 - 1

        arr_label = arr_label.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        out_dict["low_res"] = np.transpose(arr_label[:, :, None], [2, 0, 1])
        return np.transpose(arr, [2, 0, 1]), out_dict


def random_crop_arr_with_point(pil_image, image_size, crop_x, crop_y, min_crop_frac=0.8, max_crop_frac=1.0):
    # min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    # max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    # smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
    #
    # # We are not on a new enough PIL to support the `reducing_gap`
    # # argument, which uses BOX downsampling at powers of two first.
    # # Thus, we do it by hand to improve downsample quality.
    # while min(*pil_image.size) >= 2 * smaller_dim_size:
    #     pil_image = pil_image.resize(
    #         tuple(x // 2 for x in pil_image.size), resample=Image.BOX
    #     )
    #
    # scale = smaller_dim_size / min(*pil_image.size)
    # pil_image = pil_image.resize(
    #     tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    # )

    arr = np.array(pil_image)
    # crop_y = random.randrange(arr.shape[0] - image_size + 1)
    # crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def load_dwt_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(os.path.join(data_dir, 'palm'))
    all_labels = _list_image_files_recursively(os.path.join(data_dir, 'label'))

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = DwtImageDataset(
        image_size,
        all_files,
        all_labels,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class DwtImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        label_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_labels = label_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        label_path = self.local_labels[idx]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        with bf.BlobFile(label_path, "rb") as f:
            pil_label = Image.open(f)
            pil_label.load()

        pil_image = pil_image.convert("RGB")
        pil_label = pil_label.convert("RGB")

        pil_image = pil_image.resize((self.resolution * 2, self.resolution * 2), Image.BILINEAR)
        pil_label = pil_label.resize((self.resolution, self.resolution), Image.BILINEAR)

        

        # size = int(self.resolution * 1.1)
        # pil_image = pil_image.resize((size, size), Image.BILINEAR)
        # pil_label = pil_label.resize((size, size), Image.BILINEAR)

        # if self.random_crop:
        #     crop_y = random.randrange(pil_image.size[0] - self.resolution + 1)
        #     crop_x = random.randrange(pil_image.size[0] - self.resolution + 1)
        #     arr = random_crop_arr_with_point(pil_image, self.resolution, crop_x, crop_y)
        #     arr_label = random_crop_arr_with_point(pil_label, self.resolution, crop_x, crop_y)
        # else:
        #     arr = center_crop_arr(pil_image, self.resolution)
        #     arr_label = center_crop_arr(pil_label, self.resolution)

        arr = np.array(pil_image)
        arr_label = np.array(pil_label)

        if self.random_flip:
            if random.random() < 0.5:
                arr = arr[::-1, :]
                arr_label = arr_label[::-1, :]

            if random.random() < 0.5:
                arr = arr[:, ::-1]
                arr_label = arr_label[:, ::-1]
        
            if random.random() < 0.5:
                angle = np.random.randint(1, 4)
                arr = np.rot90(arr, angle)
                arr_label = np.rot90(arr_label, angle)

        arr = arr.astype(np.float32) / 127.5 - 1
        arr_label = arr_label.astype(np.float32) / 127.5 - 1


        # pil_bg = pil_label.resize((self.resolution *2 , self.resolution * 2), Image.BILINEAR)
        # arr_bg = np.array(pil_bg, dtype=np.uint8)
        # _, arr_mask = cv.threshold(cv.cvtColor(arr_bg, cv.COLOR_RGB2GRAY), 0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # arr_mask = arr_mask.astype(np.float32) / 255.0

        # arr_bg = arr_bg.astype(np.float32) / 127.5 - 1
        # arr_bg = arr_bg * arr_mask[:, : , None]
        # arr = arr - arr_bg


        coeffs = pywt.dwt2(arr, 'haar', axes=(0, 1))
        arr_A, (arr_H, arr_V, arr_D) = coeffs

        arr_A = arr_A / 2.0

        # arr_dwt = np.concatenate([arr_A[:, :, None], arr_H[:, :, None], arr_V[:, :, None], arr_D[:, :, None]], axis=-1)
        arr_dwt = np.concatenate([arr_A, arr_H, arr_V, arr_D], axis=-1)     

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        out_dict["low_res"] = np.transpose(arr_label, [2, 0, 1])
        return np.transpose(arr_dwt, [2, 0, 1]), out_dict



def load_scale_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=True,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(os.path.join(data_dir, 'palm'))
    all_labels = _list_image_files_recursively(os.path.join(data_dir, 'label'))

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = ScaleImageDataset(
        image_size,
        all_files,
        all_labels,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class ScaleImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        label_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=True,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_labels = label_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        label_path = self.local_labels[idx]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        with bf.BlobFile(label_path, "rb") as f:
            pil_label = Image.open(f)
            pil_label.load()

        pil_image = pil_image.convert("L")
        pil_label = pil_label.convert("L")

        size = int(self.resolution * 1.1)
        pil_image = pil_image.resize((size, size), Image.BILINEAR)
        pil_label = pil_label.resize((size, size), Image.BILINEAR)

        if self.random_crop:
            crop_y = random.randrange(pil_image.size[0] - self.resolution + 1)
            crop_x = random.randrange(pil_image.size[0] - self.resolution + 1)
            arr = random_crop_arr_with_point(pil_image, self.resolution, crop_x, crop_y)
            arr_label = random_crop_arr_with_point(pil_label, self.resolution, crop_x, crop_y)
        else:
            arr = center_crop_arr(pil_image, self.resolution)
            arr_label = center_crop_arr(pil_label, self.resolution)

        if self.random_flip:
            # if random.random() < 0.5:
            #     arr = arr[::-1, :]
            #     arr_label = arr_label[::-1, :]

            if random.random() < 0.5:
                arr = arr[:, ::-1]
                arr_label = arr_label[:, ::-1]
        
            # if random.random() < 0.5:
            #     angle = np.random.randint(1, 4)
            #     arr = np.rot90(arr, angle)
            #     arr_label = np.rot90(arr_label, angle)

        # arr = arr.astype(np.float32) / 127.5 - 1
        arr = arr.astype(np.float32) / 255.0
        arr = arr - np.mean(arr) 
        arr = arr * 5.0

        arr_label = arr_label.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        out_dict["low_res"] = np.transpose(arr_label[:, :, None], [2, 0, 1])
        return np.transpose(arr[:, :, None], [2, 0, 1]), out_dict
