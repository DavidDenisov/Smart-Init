# our try to augment the images, was not used since this seemed to worsen the results.
# Used to apply `torch.transpose` on the images.
import numpy as np
import torch
import tensorflow_datasets as tfds
import cv2
from time import perf_counter as time
import torchvision.transforms as transforms

interpolation=transforms.InterpolationMode("bilinear")
transform_func = transforms.AutoAugment(interpolation=interpolation)  # https://pytorch.org/vision/main/generated/torchvision.transforms.AutoAugment.html


# Our try to augment the images.
def transform_(images):
    ans = torch.zeros_like(images)
    for i in range(len(images)):
           ans[i] = transform_func(images[i])
    return ans

# Given a batch of images translates them, and if transform is true (which is not used in the paper) also augments them.
def transform(images,transform=True):
    cur = torch.tensor(np.asarray(images))

    cur = torch.transpose(cur, 1,3)
    cur = torch.transpose(cur, 2,3)

    # cur = resize_up(cur)
    if transform:
        # t = time()
        cur = transform_(cur)
        # print(np.round(time()-t,5))
    return np.array(cur)


# A function to validate the augmentation.
if __name__ == '__main__':
    ds_train = tfds.data_source('imagenet_resized/64x64', data_dir='./data/tfds', split='train')
    i = 0
    images = []
    for val in ds_train:
        images.append(val['image'])
        i += 1
        if i > pow(2,10):
            break
    a = transform(images)
    b = transform(images,False)

    for i in range(100):
        cv2.imshow('img1',b[i,0])
        cv2.waitKey(1)

        cv2.imshow('img2',a[i,0])
        c = cv2.waitKey(0)
    #   
    #   cv2.imshow('img2',images_[0])
    #   cv2.waitKey(0)
    print(1)
