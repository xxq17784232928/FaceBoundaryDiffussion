"""
    Refer to https://github.com/rosinality/stylegan2-pytorch/blob/master/prepare_data.py
"""

import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import os, glob, sys

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, resample, quality=100):
    # img = trans_fn.resize(img, (size, size), resample)
    img = trans_fn.resize(img, (size, size), interpolation=resample)
    # img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file, img_id = img_file
    # print("check resize_worker:", i, file, img_id)
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out, img_id


def file_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
    files = [f.rstrip() for f in files]
    return files



def prepare(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)
    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file, file.split('/')[-1].split('.')[0]) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs, img_id in tqdm(pool.imap_unordered(resize_fn, files)):
            key_label = f"{str(i).zfill(5)}".encode("utf-8")
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")
                with env.begin(write=True) as txn:
                    txn.put(key, img)
                    txn.put(key_label, str(img_id).encode("utf-8"))

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

def get_attribute_index(attr_name, attr_file='/hexp/xxq/project/BoundaryDiffusion/utils/celeba_attr.txt'):
    # 从文件中读取属性列表
    try:
        with open(attr_file, 'r', encoding='utf-8') as f:
            attr_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"属性文件 '{attr_file}' 未找到。")
    
    # 查找属性名称的索引
    try:
        index = attr_list.index(attr_name)
        return index
    except ValueError:
        raise ValueError(f"属性 '{attr_name}' 不存在于属性列表中。")


def prepare_attr(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, label_attr='gender'
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)
    files = sorted(dataset.imgs, key=lambda x: x[0])
    attr_file_path = '/hexp/xxq/project/BoundaryDiffusion/output/lmdb/list_attr_celeba.txt'
    labels = file_to_list(attr_file_path)
    attr_dict = {}
    files_attr = []

    # 获取 'Smiling' 属性的索引
    index_smiling = get_attribute_index(label_attr)
    print(f"'{label_attr}' 的索引是 {index_smiling}")

    for i, (file, split) in enumerate(files):
        img_id = int(file.split('/')[-1].split('.')[0])
        attr_label = labels[img_id+1].split()
        # label = int(attr_label[21])
        label = int(attr_label[index_smiling+1])
        # files_attr.append((i, file, label))
        # files_attr.append((i, file, label))
        files_attr.append((i, file, img_id))

    files = files_attr
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs, label in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")
                key_label = f"{'label'}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)
                    txn.put(key_label, str(label).encode("utf-8"))

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))
            txn.put("index_attr".encode("utf-8"),str(index_smiling+1).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str)
    parser.add_argument("--size", type=str, default="128,256,512,1024")
    parser.add_argument("--n_worker", type=int, default=5)
    parser.add_argument("--resample", type=str, default="bilinear")
    parser.add_argument("--attr", type=str)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]
    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare_attr(env, imgset, args.n_worker, sizes=sizes, resample=resample,label_attr=args.attr)