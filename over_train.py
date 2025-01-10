from main import optimizer_constructor, print_progress_bar
import gc
import torch
from time import perf_counter as time
import numpy as np
import sys
import torch.nn as nn
import torch.utils.data as data_utils
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from Mobile import MobileNetV1 as MobileNet

# Generates the results for Figure 5 in the paper.

# Note that due to the use of global parameters, and to allow easier modification for specific tests,
# all the functions beside calc_res have parallel functions in cifar and main.

# Set up the device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up global training parameters.
epochs = 75
criterion = nn.CrossEntropyLoss().to(device)

# Set up sample, train, and test loaders.
# We train only on smpl_loader, and the rest are used for testing.
ds_train = torch.load('./data/image_net/32/train.pt')

ds_test = torch.load('./data/image_net/32/test.pt')

smpl_loader = data_utils.DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=0)
train_loader = data_utils.DataLoader(ds_train, batch_size=1024, shuffle=False, num_workers=0)
test_loader = data_utils.DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=0)


# Function to sample a random subset from smpl_loader.
# Since the data is loaded randomly, take up to up_to batches from smpl_loader.
def sample_set(up_to):
    dataset = []
    for i, (images, labels) in enumerate(smpl_loader):
        if i < up_to:
            dataset.append((images, labels))
        else:
            break
    return dataset


def calc_res(dataset, model):
    with torch.no_grad():
        correct = 0
        total = 0
        loss = []
        t = time()
        for i,(images, labels) in enumerate(dataset):
            print_progress_bar(i, len(dataset), 'Test', time() - t)
            images, labels = images.to(device=device).to(dtype=torch.float), labels.to(device=device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.append(criterion(outputs, labels).cpu().data.numpy().item())
        sys.stdout.write("\r")
        # print('Took', np.round(time() - t, 3))
    a,b = 100 * correct / total, np.mean(loss)
    print(np.round(a,3),np.round(b,3))
    return a,b


# Function to run the training on a subsample from the training set, since we do not run training from the result,
# this is used as a run_one_test.
def find_mean_on_sample(train_samp):
    # Construct model, optimizer, scheduler, and sample a subset from the training set to train on.
    model = MobileNet(ch_in=3, n_classes=1000).to(device)
    opt, sec = optimizer_constructor(model)
    dataset = sample_set(train_samp)

    # Train the model on the sample taken, and save the results for the sample set, test set, and whole training set.
    smp, val, train = [], [], []
    smp.append(calc_res(dataset, model))
    val.append(calc_res(test_loader, model))
    train.append(calc_res(train_loader, model))
    for step in range(epochs):
        t = time()
        # Run a training epoch.
        model.train()
        for i, (images, labels) in enumerate(dataset):
            print_progress_bar(i, len(dataset), 'Train', time() - t)
            images, labels = images.to(device=device).to(dtype=torch.float), labels.to(device=device)
            # Forward pass
            outputs = model(images)
            loss1 = criterion(outputs, labels)

            # Backward and optimize
            opt.zero_grad()
            loss1.backward()
            opt.step()

        sys.stdout.write("\r")

        model.eval()
        sec.step()
        # Evaluate the results over the sample set, test set, and whole training set.
        with torch.no_grad():
            smp.append(calc_res(dataset, model))
            val.append(calc_res(test_loader, model))
            train.append(calc_res(train_loader, model))

        print('Step ' + str(step + 1) + ' done:')
        print('Took', np.round(time() - t, 3))
    del dataset
    gc.collect()
    return smp, val, train


# Run a whole experiment from the paper, modify the hard coded values to change which one.
def run():
    if device == 'cuda':
        cudnn.benchmark = True

    S, V, T = [], [], []
    for i in range(4):
        print('--------------------------------------')
        print('--------------------------------------')
        print('Run',i,'starts:')
        print('Run', i, 'starts:')
        print('Run', i, 'starts:')
        print('--------------------------------------')
        print('--------------------------------------')
        s, v, t = find_mean_on_sample(10_000)
        S.append(s), V.append(v), T.append(t)
    np.savez('./res_over', S=np.array(S), V=np.array(V), T=np.array(T))


# Plot the results, done here since it is slightly different from for the other tests.
def plot():
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.rcParams.update({'font.size': 50})
    plt.rcParams.update({'lines.linewidth': 25})

    # Extract the values saved.
    dict = np.load('res_32_on_sample.npz')
    S, V, T = [dict[l] for l in ['S', 'V', 'T']]

    n = S[0].shape[0]
    R = np.arange(n)

    names = {'Sample': (S, 'r'), 'Train': (T, 'b'), 'Validation': (V, 'g')}
    measures = ['Accuracy', 'Loss']
    # Plot each measurement, for each set from the sample, training, and test.
    for i, mesure in enumerate(measures):
        for name in names.keys():
            cur, c = names[name]
            cur = cur[:,:,i]
            s1, s2 = np.percentile(cur, 25, 0), np.percentile(cur, 75, 0)

            m = np.median(cur, 0)

            plt.plot(R, m, c=c, label=name, ls='--')
            plt.fill_between(R, s1, s2, alpha=0.25, color=c, edgecolor=None)

        plt.xlabel('Epoch')
        plt.ylabel(mesure)
        plt.legend()
        # Optional logarithmic scale, since in some instances the values increase extremely.

        plt.yscale('log',base=2)
        # plt.show()
        plt.savefig('./Figs/' + mesure + '.png', dpi=250, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    # run()
    plot()
