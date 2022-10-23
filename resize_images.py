import os
import torch
import albumentations
import pretrainedmodels
import numpy as np
import pandas as pd
import torch.nn as nn

from apex import amp
from sklearn import metrics
from torch.nn import functional as F


from wtfml.data_loaders.image import ClassificationLoader 
from wtfml.engine import Engine
from wtfml.utils import EarlyStopping


# resize images
def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )
    img.save(outpath)


# resize train images
input_folder = "C:\\Users\\neamul\\Dataset_Melanoma Detection\\jpeg\\train"
output_folder = "C:\\Users\\neamul\\Dataset_Output\\train"
images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(n_jobs=12) (
    delayed(resize_image)(
        i, output_folder, (512, 512)
    ) for i in tqdm(images)
)


# resize test images
input_folder = "C:\\Users\\neamul\\Dataset_Melanoma Detection\\jpeg\\test"
output_folder = "C:\\Users\\neamul\\Dataset_Output\\test"
images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(n_jobs=12) (
    delayed(resize_image)(
        i, output_folder, (512, 512)
    ) for i in tqdm(images)
)