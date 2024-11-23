"""Crop l8 biome dataset (download from torchgeo) into 512x512 patches"""
import glob
import os
import re
import numpy as np
import tifffile
from tqdm import tqdm


def get_filenames(root, pattern):
    return glob.glob(f"{root}/**/{pattern}", recursive=True)


def read_and_process_image(filepath, ann_filepath):
    img = tifffile.imread(filepath)
    ann = tifffile.imread(ann_filepath)
    ann[ann == 64] = 1
    ann[ann == 128] = 0
    ann[ann == 192] = 2
    ann[ann == 255] = 3
    return img, ann


def save_patches(img, ann, subdir, subsubdir):
    patch_size = 512
    stride = 512

    for i in range(0, img.shape[0], stride):
        for j in range(0, img.shape[1], stride):
            img_patch = img[i:i + patch_size, j:j + patch_size]
            ann_patch = ann[i:i + patch_size, j:j + patch_size]

            if img_patch.shape[0] == patch_size and img_patch.shape[1] == patch_size:
                img_patch_filename = f"data/l8/img_dir/{subdir}_{subsubdir}_patch_{i}_{j}.TIF"
                ann_patch_filename = f"data/l8/ann_dir/{subdir}_{subsubdir}_patch_{i}_{j}.TIF"

                if np.unique(ann_patch).any() != 0:
                    tifffile.imsave(img_patch_filename, img_patch)
                    tifffile.imsave(ann_patch_filename, ann_patch)


def process_directory(root, filename_regex):
    for subdir in tqdm(os.listdir(root)):
        for subsubdir in os.listdir(os.path.join(root, subdir)):
            for filename in os.listdir(os.path.join(root, subdir, subsubdir)):
                if re.match(filename_regex, filename, re.VERBOSE):
                    filepath = os.path.join(root, subdir, subsubdir, filename)
                    ann_filepath = filepath.replace('.TIF', '_fixedmask.TIF')
                    img, ann = read_and_process_image(filepath, ann_filepath)
                    save_patches(img, ann, subdir, subsubdir)


if __name__ == "__main__":
    root = "data/l8_biome"
    filename_glob = "LC8*.TIF"
    filename_regex = r"""
        ^LC8
        (?P<wrs_path>\d{3})
        (?P<wrs_row>\d{3})
        (?P<date>\d{7})
        (?P<gsi>[A-Z]{3})
        (?P<version>\d{2})
        \.TIF$
    """
    process_directory(root, filename_regex)
