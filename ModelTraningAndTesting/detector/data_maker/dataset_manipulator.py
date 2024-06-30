"""
Splits  a monolithic dataset into train-test or train-va-test

@author Lucian Sasu lmsasu@unitbv.ro
"""
import logging
import os.path
import shutil
from typing import Tuple, List

import cv2
import numpy as np

import splitfolders
from tqdm import tqdm

from utils.log_utils import setup_logger
from utils.timing import timeit


# source: https://medium.com/@gavin_xyw/letterbox-in-object-detection-77ee14e5ac46
def __resize__(img, new_size, letter_box=True):
    """img:         input image in numpy array
       new_size: [height, width] of input image, this is the target shape for the model
       letter_box:  control whether to apply letterbox resizing """
    if letter_box:
        img_h, img_w, _ = img.shape                    #img is opened with opencv, in shape(h, w, c), this is the original image shape
        new_h, new_w = new_size[0], new_size[1]  # desired input shape for the model
        offset_h, offset_w = 0, 0                      # initialize the offset
        if (new_w / img_w) <= (new_h / img_h):         # if the resizing scale of width is lower than that of height
            new_h = int(img_h * new_w / img_w)         # get a new_h that is with the same resizing scale of width
            offset_h = (new_size[0] - new_h) // 2   # update the offset_h
        else:
            new_w = int(img_w * new_h / img_h)         # if the resizing scale of width is higher than that of height, update new_w
            offset_w = (new_size[1] - new_w) // 2   # update the offset_w
        resized = cv2.resize(img, (new_w, new_h))      # get resized image using new_w and new_h
        img = np.full((new_size[0], new_size[1], 3), 127, dtype=np.uint8) # initialize a img with pixel value 127, gray color
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (new_size[1], new_size[0]))

    return img

@timeit
def resize_split_data(input_path: str,
                      resized_path: str,
                      output_path: str,
                      new_size: Tuple[int, int],
                      min_images: int,
                      ratio: Tuple[float, float] | Tuple[float, float, float],
                      seed: int,
                      move: bool = False
                      ) -> Tuple[str, List[str]]:
    assert os.path.isdir(input_path), f'Directory {input_path} does not exist'
    lst_new_dirs = resize_images(input_path, resized_path, new_size, min_images=min_images)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    splitfolders.ratio(resized_path, output=output_path, seed=seed, ratio=ratio, group_prefix=None, move=move)
    return output_path, lst_new_dirs

@timeit
def resize_images(input_path: str,
                  resized_path: str,
                  new_size: Tuple[int, int],
                  min_images: int=20) -> List[str]:
    if os.path.exists(resized_path):
        shutil.rmtree(resized_path)
    os.makedirs(resized_path)

    lst_dirs = []

    for root, dirs, files in tqdm(os.walk(input_path), initial=1):
        if len(files) < min_images:
            logging.info(f'Directory {root} contains {len(files)}<{min_images} images, skipped')
            continue

        new_path = os.path.join(resized_path, os.path.basename(root))
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        os.makedirs(new_path)
        lst_dirs.append(new_path)

        logging.info(f'Will resize to {new_size} in {root} {len(files)} files')
        for file in files:
            original_img_path = os.path.join(root, file)
            resized_img_path = os.path.join(new_path, file)
            img = cv2.imread(original_img_path)
            letterboxed = __resize__(img, new_size, letter_box=True)
            cv2.imwrite(resized_img_path, letterboxed)

    return lst_dirs


if __name__ == '__main__':
    caller_module: str = __file__.split(os.sep)[-1].split('.')[0]
    setup_logger(caller_module)

    new_size: Tuple[int, int] = (224, 224)
    min_images: int = 40
    logging.info(f'{new_size=}\t{min_images}')

    input_path: str = '../data/traffic_Data/DATA'
    resized_path: str = f'../data/traffic_Data/DATA_{new_size[0]}x{new_size[1]}'
    output_path: str = '../../data/traffic_Data/split'
    logging.info(f'{input_path=}\n{resized_path=}\n{output_path}')

    output_path, kept_directories = resize_split_data(input_path=input_path,
                                                      resized_path=resized_path,
                                                      output_path=output_path,
                                                      new_size=new_size,
                                                      min_images=min_images,
                                                      ratio=(0.8, 0.2),
                                                      seed=42,
                                                      move=False)

    logging.info(f'Directories kept: {len(kept_directories)}')
