import gc
import torch
import torch.utils.data as data_utils
from image_augment import transform
from time import perf_counter as time
import tensorflow_datasets as tfds
import tensorflow
import numpy as np
import sys


# Code to extract imagenet from tfds and transform it to torch tensor. This allows easier integration with PyTorch.
# To modify the extracted datasets and their location, change the hard coded values.

# Helper function to print a progression bar with time.
def print_progress_bar(index, total, label, time_taken):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}  took {np.round(time_taken, 1)}")
    sys.stdout.flush()


# Extract the training dataset, possibly with ineq that allows extracting only a label-based part of it to preserve ram.
def extract_train(ineq, s):
    ds_train = tfds.data_source('imagenet_resized/16x16', data_dir='../data/tfds', split='train')
    # transform_func = torchvision.transforms.AutoAugment()  # https://pytorch.org/vision/main/generated/torchvision.transforms.AutoAugment.html

    images_cur, images, labels = [], [], []
    i = 0
    n = len(ds_train)
    t = time()
    for val in ds_train:
        print_progress_bar(i, n, 'load train', time() - t)
        i += 1

        image, label = val['image'], val['label']
        # Change to extract only the images in the dataset where the ineq holds.
        # if ineq(label):
        images_cur.append(image)
        labels.append(label)

        # If there are enough images in the buffer, translate them in a batch.
        if len(images_cur) > pow(2, 14):
            cur = transform(images_cur, False)

            images.extend(cur)

            del images_cur, cur
            gc.collect()

            images_cur = []

    # Translate the remaining images.
    if len(images_cur) > 0:
        cur = transform(images_cur, False)
        images.extend(cur)

        del images_cur, cur
        gc.collect()

    print('\r')
    print('train part' + s + ' loaded')

    # Modify the labels and images to be torch tensors.
    labels = np.asarray(labels)

    gc.collect()

    labels = torch.as_tensor(labels)

    gc.collect()

    images = np.array(images)
    images = torch.as_tensor(images)

    # Save the results in a temporary file.
    torch.save(images, './temp_images' + s + '.pt')
    torch.save(labels, './temp_labels' + s + '.pt')

    print('train part ' + s + ' extracted')


# Extract the test dataset.
def extract_test():
    ds_test = tfds.data_source('imagenet_resized/16x16', data_dir='../data/tfds', split='validation')

    # Extract the test dataset to python lists.
    images, labels = [], []
    i = 0
    n = len(ds_test)
    t = time()
    for val in ds_test:
        i += 1
        print_progress_bar(i, n, 'load test', time() - t)
        image, label = val['image'], val['label']
        # if label<100:
        labels.append(label)
        images.append(image)

    # Modify the labels and images to be torch tensors.
    labels = np.asarray(labels)
    images = np.asarray(images)

    labels = torch.as_tensor(labels)
    images = torch.as_tensor(images)

    images = torch.transpose(images, 1, 3)
    images = torch.transpose(images, 2, 3)

    # Save the results.
    test = data_utils.TensorDataset(images, labels)

    torch.save(test, './image_net_16x16_val.pt')
    print('\r')
    print('test saved')

    del images, labels, test
    gc.collect()


# Code that extracts the training dataset, optionally in two temp files to preserve ram.
# For images of size 64x64 this ram reduction can be necessary if the machine has less than 64GB of ram.
# Otherwise, it is possible to create a swap file to bump the memory.
def extract_data_part1():
    def ineq(label):
        if label < 500:
            return True
        return False

    extract_train(ineq, '_1')

    #   def ineq(label):
    #       if 500<=label:
    #           return True
    #       return False
    #
    #   extract_train(ineq, '_2')


# Code that saves the training dataset, after it being extracted to temporary files.
# With optional (commented) support to use the 2 temporary files.
def extract_data_part2():
    print('Saving train')

    labels1, images1 = torch.load('./temp_labels_1.pt'), torch.load('./temp_images_1.pt')
    gc.collect()

    #   gc.collect()
    #   labels1, labels2 = torch.load('./temp_labels_1.pt'), torch.load('./temp_labels_2.pt')
    #   gc.collect()
    #   images1, images2 = torch.load('./temp_images_1.pt'), torch.load('./temp_images_2.pt')
    #   gc.collect()
    #   labels1 = torch.concatenate([labels1, labels2], 0)
    #   gc.collect()
    #   images1 = torch.concatenate([images1, images2], 0)
    #   gc.collect()

    train = data_utils.TensorDataset(images1, labels1)
    torch.save(train, './image_net_16x16_train.pt')
    print('Train saved')


if __name__ == '__main__':
    extract_test()
    extract_data_part1()
    extract_data_part2()
