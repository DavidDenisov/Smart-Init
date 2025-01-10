import glob
import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data_utils
from time import perf_counter as time
from PIL import Image
import json

def print_progress_bar(index, total, label, time_taken):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}  took {np.round(time_taken, 1)}")
    sys.stdout.flush()


f = open('./image_net_v2_meta.json')
v2_json = json.load(f)
v2 = []
for arg in v2_json:
    v2.append(arg['wnid'])

f = open('./imagenet_resized_labels.txt')
arr = f.read().splitlines()

T = [arr.index(val) for val in v2]

images,labels = [],[]
total = 10_000
t = time()
# res = np.zeros((10,224,224,3))
for c_path in glob.iglob('./imagenet_test_v2/*'):
    c = str.split(c_path,'/')[-1]
    label = int(c)
    label = T[label]
    for im_path in glob.iglob(c_path+'/*.jpeg'):
        print_progress_bar(len(labels),total,'Load images',time()-t)
        image = Image.open(im_path)

        image = image.resize((16,16), Image.LANCZOS)
        image = np.array(image)

        images.append(image)
        labels.append(label)


# Modify the labels and images to be torch tensors.
labels = np.asarray(labels)
images = np.asarray(images)

labels = torch.as_tensor(labels)
images = torch.as_tensor(images)

images = torch.transpose(images, 1, 3)
images = torch.transpose(images, 2, 3)

test = data_utils.TensorDataset(images, labels)

torch.save(test, './test_16.pt')