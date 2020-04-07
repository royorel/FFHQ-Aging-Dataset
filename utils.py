# Copyright (c) 2020, Roy Or-El. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# This code is a modification of the utils.py file
# from the https://github.com/chenxi116/DeepLabv3.pytorch repository


import os
import math
import html
import glob
import uuid
import random
import hashlib
import requests
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def preprocess_image(image, flip=False, scale=None, crop=None):
    if flip:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if scale:
        w, h = image.size
        rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        image = image.resize(new_size, Image.ANTIALIAS)

    data_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = data_transforms(image)

    return image


def download_file(session, file_spec, chunk_size=128, num_attempts=10):
    file_path = file_spec['file_path']
    file_url = file_spec['file_url']
    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            break

        except:
            # Last attempt => raise error.
            if not attempts_left:
                raise

            # Handle Google Drive virus checker nag.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                links = [html.unescape(link) for link in data.decode('utf-8').split('"') if 'export=download' in link]
                if len(links) == 1:
                    file_url = requests.compat.urljoin(file_url, links[0])
                    continue

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass
