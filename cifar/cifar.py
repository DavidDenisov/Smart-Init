import torch
import gc
from time import perf_counter as time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from copy import deepcopy
import numpy as np
import torchvision
import torchvision.transforms as transforms
from main import calc_accuracy, print_progress_bar

from Mobile import MobileNetV1 as MobileNet
import sys

# Contains the tests done for cifar-10, to change between the tests modify directly.

# Note that due to the use of global parameters, and to allow easier modification for specific tests,
# all the functions have similar and parallel functions in main and over_train.

# Set up the device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up global training parameters.
lr = 1e-3
num_epochs = 100
criterion = nn.CrossEntropyLoss().to(device)
batch_size = 32

from cifar_data_loader import get_train_valid_loader, get_test_loader

train_set, val_set = get_train_valid_loader(data_dir='../data/cifar', batch_size=batch_size)
test_set = get_test_loader(data_dir='../data/cifar', batch_size=batch_size)


#   # Set up augmenting function.
#   transform_train = transforms.Compose([
#       transforms.ToTensor(),
#       transforms.Resize((64,64),antialias=False),
#       transforms.Normalize((0,0,0), (1,1,1))
#   ])
#
#   transform_test = transforms.Compose([
#       transforms.ToTensor(),
#       transforms.Resize((64,64),antialias=False),
#       transforms.Normalize((0,0,0), (1,1,1))
#   ])
#
#   # Set up train and test loaders.
#   train_subset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True,
#                                           download=True, transform=transform_train)
#
#   # val_subset = torchvision.datasets.CIFAR10(root='./data/cifar', split="valid",
#   #                                        download=True, transform=transform_train,target_type='identity')
#
#   testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False,
#                                          download=True, transform=transform_test)
#
#   train_set = torch.utils.data.DataLoader(dataset=train_subset, shuffle=True, batch_size=batch_size, num_workers=2)
#   #val_set = torch.utils.data.DataLoader(dataset=val_subset, shuffle=False, batch_size=batch_size, num_workers=2)
#
#   test_set = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)


# Helper function to construct optimizer and scheduler for a model.
def optimizer_constructor(model):
    # total training epoches
    #   MILESTONES = [60, 120, 160]
    #   optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    #   train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES,gamma=0.2)
    # lr=0.1 # got around 50.5 val-accuracy
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.925)
    return optimizer, scheduler


# Function to sample a random subset from train_loader.
# Since the data is loaded randomly, take up to up_to batches from train_loader.
def sample_set(up_to):
    dataset = []
    for i, (images, labels) in enumerate(train_set):
        if i < up_to:
            dataset.append((images, labels))
        else:
            break
    return dataset


# Function to run the suggested initialization with predefined parameters.
def find_mean_on_sample(model, criterion, train_samp, T_max=10):
    # Construct optimizer, scheduler, and sample a subset from the training set to train on.
    opt, sec = optimizer_constructor(model)
    dataset = sample_set(train_samp)

    # Train the model on the sample taken.
    prev_acc = calc_accuracy(dataset, model)
    for step in range(T_max):
        model.train()
        new_acc = 0
        t = time()
        for i, (images, labels) in enumerate(dataset):
            print_progress_bar(i, len(dataset), 'Pre-train smart init', time() - t)
            images, labels = images.to(device=device).to(dtype=torch.float), labels.to(device=device)
            # Forward pass
            outputs = model(images)
            loss1 = criterion(outputs, labels)

            # Backward and optimize
            opt.zero_grad()
            loss1.backward()
            opt.step()

            # Update the accuracy.
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            new_acc += (predicted == labels).sum().item() / total

        new_acc = 100 * new_acc / train_samp
        model.eval()
        sec.step()

        # Print the accuracy change and the time the epoch took.
        sys.stdout.write("\r")
        print('Took', np.round(time() - t, 3))

        print('Step ' + str(step + 1) + ' done: ', np.round(prev_acc, 2), np.round(new_acc, 2))
        prev_acc = new_acc

    del dataset
    gc.collect()
    return model
    # return net_opt


# Runs the suggested initialization, along with debugging prints.
def smart_init(model, criterion):
    sample = 141
    # Print the percentage of the sample taken from the train set.
    print(np.round(100 * sample / len(train_set), 2))
    print(len(train_set))
    old = calc_accuracy(val_set, model)

    # Run the suggested initialization.
    t = time()
    model = find_mean_on_sample(model, criterion, 14, T_max=10)
    model = find_mean_on_sample(model, criterion, sample, T_max=10)
    # model = find_mean_on_sample(model, criterion, 780,10)
    t = time() - t

    # Print the time the initialization took along with the accuracy change.
    print('Smart init done: took', np.round(t, 2), 's.')

    new = calc_accuracy(val_set, model)
    print('Started with accuracy', np.round(old, 2), 'ended with accuracy', np.round(new, 2), '.')

    return t, model


# Runs the warm start initialization, along with debugging prints.
def warm_start_init(model, criterion):
    sample = 704
    # Print the percentage of the sample taken from the train set.
    print(np.round(100 * sample / len(train_set), 2))
    print(len(train_set))
    old = calc_accuracy(val_set, model)

    # Run the suggested initialization.
    t = time()
    model = find_mean_on_sample(model, criterion, sample, 50)
    t = time() - t

    # Print the time the initialization took along with the accuracy change.
    print('Smart init done: took', np.round(t, 2), 's.')

    new = calc_accuracy(val_set, model)
    print('Started with accuracy', np.round(old, 2), 'ended with accuracy', np.round(new, 2), '.')

    return t, model


# Run one of the tests mentioned in the paper, i.e., initialize a network in two ways and train both in parallel.
def one_run():
    model = MobileNet(ch_in=3, n_classes=10)

    comp = model.to(device)
    smart = deepcopy(comp).to(device)  # Copy the model to use same weights
    warm = deepcopy(comp).to(device)  # Copy the model to use same weights
    # Validate the same values, optional.

    #   diff = np.abs(comp.get_weight() - our.get_weight()).sum().item()
    #   print(diff)

    if device == 'cuda':
        cudnn.benchmark = True

    # Initialize our network.
    init_time, smart = smart_init(smart, criterion)
    # Initialize warm start.
    init_time, warm = warm_start_init(warm, criterion)
    # Set up optimizers and schedulers.
    smart_opt, smart_sce = optimizer_constructor(smart)
    comp_opt, comp_sce = optimizer_constructor(comp)
    warm_opt, warm_sce = optimizer_constructor(warm)

    # Set up results arrays, values would be overwritten, allows noticing bugs of missing tests, e.t.c.
    train_losses = np.ones(num_epochs) * np.inf, np.ones(num_epochs) * np.inf, np.ones(num_epochs) * np.inf
    train_accuracy = -np.ones(num_epochs), -np.ones(num_epochs), -np.ones(num_epochs)
    val_losses = np.ones(num_epochs) * np.inf, np.ones(num_epochs) * np.inf, np.ones(num_epochs) * np.inf
    val_accuracy = -np.ones(num_epochs), -np.ones(num_epochs), -np.ones(num_epochs)
    time_from_best_comp = 0
    best_accuracy1, best_accuracy2, best_accuracy3 = 0, 0, 0
    # Run the training on both models.
    for epoch in range(num_epochs):
        t = time()

        s_l, c_l, w_l, s_ac, c_ac, w_ac = 0, 0, 0, 0, 0, 0

        # Train both models.
        gc.collect()
        smart.train()
        warm.train()
        comp.train()
        t_c = time()
        for i, (images, labels) in enumerate(train_set):
            # print(np.round(time()-t___,3))
            total = labels.size(0)

            print_progress_bar(i, len(train_set), 'Train', time() - t_c)

            images, labels = images.to(device=device).to(dtype=torch.float), labels.to(device=device)

            # Run for comp.

            # Forward pass
            outputs = comp(images)
            loss1 = criterion(outputs, labels)

            # Backward and optimize
            comp_opt.zero_grad()
            loss1.backward()
            comp_opt.step()

            c_l += loss1.cpu().data.numpy().item()
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            c_ac += (predicted == labels).sum().item() / total

            # Run for smart init.

            # Forward pass
            outputs = smart(images)
            loss_smart = criterion(outputs, labels)

            # Backward and optimize
            smart_opt.zero_grad()
            loss_smart.backward()
            smart_opt.step()

            s_l += loss_smart.cpu().data.numpy().item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            s_ac += (predicted == labels).sum().item() / total

            # Run for warm start.

            # Forward pass
            outputs = warm(images)
            loss_warm = criterion(outputs, labels)

            # Backward and optimize
            warm_opt.zero_grad()
            loss_warm.backward()
            warm_opt.step()

            w_l += loss_warm.cpu().data.numpy().item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            w_ac += (predicted == labels).sum().item() / total

        # Change the accuracy to percentage from the train set.
        s_ac, c_ac, w_ac = 100 * s_ac / len(train_set), 100 * c_ac / len(train_set), 100 * w_ac / len(train_set)
        s_l, c_l, w_l = s_l / len(train_set), c_l / len(train_set), w_l / len(train_set)

        smart_sce.step()
        warm_sce.step()
        comp_sce.step()

        #   # Print training set results.
        print('\r')
        #   print('Original train loss', np.round(c_l, 4), 'acc', np.round(c_ac, 2), '%; lr-div:',
        #         np.round(comp_opt.param_groups[0]['lr'] / lr, 4))
        #   print('Smart-Init train loss', np.round(s_l, 4), 'acc', np.round(s_ac, 2), '%; lr-div:',
        #         np.round(warm_opt.param_groups[0]['lr'] / lr, 4))
        #   print('Warm start train loss', np.round(w_l, 4), 'acc', np.round(w_ac, 2), '%; lr-div:',
        #         np.round(warm_opt.param_groups[0]['lr'] / lr, 4))

        # Save the training results.
        train_accuracy[0][epoch] = c_ac
        train_accuracy[1][epoch] = s_ac
        train_accuracy[2][epoch] = w_ac

        train_losses[0][epoch] = c_l
        train_losses[1][epoch] = s_l
        train_losses[2][epoch] = w_l

        # Test the model
        gc.collect()
        smart.eval()
        warm.eval()
        comp.eval()

        with torch.no_grad():
            total_s, total_w, total_o = 0, 0, 0
            correct_s, correct_w, correct_o = 0, 0, 0
            cur_test_losses = [], [], []
            # Compute validation loss and accuracy for both models.
            for images, labels in val_set:
                images, labels = images.to(device=device).to(dtype=torch.float), labels.to(device=device)

                # Run for comp.
                outputs = comp(images)
                _, predicted = torch.max(outputs.data, 1)
                total_o += labels.size(0)
                correct_o += (predicted == labels).sum().item()
                cur_test_losses[0].append(criterion(outputs, labels).cpu().data.numpy().item())

                # Run for smart-init.
                outputs = smart(images)
                _, predicted = torch.max(outputs.data, 1)
                total_s += labels.size(0)
                correct_s += (predicted == labels).sum().item()
                cur_test_losses[1].append(criterion(outputs, labels).cpu().data.numpy().item())

                # Run for warm-start.
                outputs = warm(images)
                _, predicted = torch.max(outputs.data, 1)
                total_w += labels.size(0)
                correct_w += (predicted == labels).sum().item()
                cur_test_losses[2].append(criterion(outputs, labels).cpu().data.numpy().item())

            # Save the validation results.
            val_losses[0][epoch] = np.mean(cur_test_losses[0])
            val_losses[1][epoch] = np.mean(cur_test_losses[1])
            val_losses[2][epoch] = np.mean(cur_test_losses[2])

            val_accuracy[0][epoch] = 100 * correct_o / total_o
            val_accuracy[1][epoch] = 100 * correct_s / total_s
            val_accuracy[2][epoch] = 100 * correct_w / total_w

            # Print the results for both methods, along with (improvement) notice.
            if best_accuracy1 >= correct_o / total_o:
                time_from_best_comp += 1
                print('Original accuracy: {} % Best: {} %'.format(100 * correct_o / total_o, 100 * best_accuracy1))
            else:
                time_from_best_comp = 0
                best_accuracy1 = correct_o / total_o
                print('Original accuracy: {} % (improvement)'.format(100 * correct_o / total_o))
            print('Original loss:', np.round(np.mean(cur_test_losses[0]), 4))

            if best_accuracy2 >= correct_s / total_s:
                print('Smart-init accuracy: {} % Best: {} %'.format(100 * correct_s / total_s, 100 * best_accuracy2))
            else:
                best_accuracy2 = correct_s / total_s
                print('Smart-init Accuracy: {} % (improvement)'.format(100 * correct_s / total_s))
            print('Smart-init loss:', np.round(np.mean(cur_test_losses[1]), 4))

            if best_accuracy3 >= correct_w / total_w:
                print('Warm-start accuracy: {} % Best: {} %'.format(100 * correct_w / total_w, 100 * best_accuracy3))
            else:
                best_accuracy3 = correct_w / total_w
                print('Warm-start Accuracy: {} % (improvement)'.format(100 * correct_w / total_w))
            print('Warm-start loss:', np.round(np.mean(cur_test_losses[2]), 4))

            print('Epoch number', epoch + 1, 'took:', np.round(time() - t, 2), 's ')

    # For good measure.
    train_losses = np.array(train_losses)
    train_accuracy = np.array(train_accuracy)
    val_losses = np.array(val_losses)
    val_accuracy = np.array(val_accuracy)

    # Test on the test set.
    gc.collect()
    smart.eval()
    warm.eval()
    comp.eval()

    with torch.no_grad():
        total_s, total_w, total_o = 0, 0, 0
        correct_s, correct_w, correct_o = 0, 0, 0
        test_losses = [], [], []
        # Compute validation loss and accuracy for both models.
        for images, labels in test_set:
            images, labels = images.to(device=device).to(dtype=torch.float), labels.to(device=device)

            # Run for comp.
            outputs = comp(images)
            _, predicted = torch.max(outputs.data, 1)
            total_o += labels.size(0)
            correct_o += (predicted == labels).sum().item()
            test_losses[0].append(criterion(outputs, labels).cpu().data.numpy().item())

            # Run for smart-init.
            outputs = smart(images)
            _, predicted = torch.max(outputs.data, 1)
            total_s += labels.size(0)
            correct_s += (predicted == labels).sum().item()
            test_losses[1].append(criterion(outputs, labels).cpu().data.numpy().item())

            # Run for warm-start.
            outputs = warm(images)
            _, predicted = torch.max(outputs.data, 1)
            total_w += labels.size(0)
            correct_w += (predicted == labels).sum().item()
            test_losses[2].append(criterion(outputs, labels).cpu().data.numpy().item())

        # Save the test results.
        comp_test_loss = [np.mean(test_losses[0])]
        smart_test_loss = [np.mean(test_losses[1])]
        warm_test_loss = [np.mean(test_losses[2])]

        comp_test_acc = [100 * correct_o / total_o]
        smart_test_acc = [100 * correct_s / total_s]
        warm_test_acc = [100 * correct_w / total_w]

    return (train_losses, train_accuracy, val_losses, val_accuracy,
            [comp_test_loss, smart_test_loss, warm_test_loss],
            [comp_test_acc, smart_test_acc, warm_test_acc])


# Run a whole experiment from the paper, modify the hard coded values to change which one.
def re_run(n):
    t = time()
    train_losses, train_accuracy, val_losses, val_accuracy, test_losses, test_accuracy = [], [], [], [], [], []
    for i in range(n):
        if i > 0:
            print('-------------------------------------------------------------------------------------------------')
            print('Done on run number ', i, 'out of', n, 'which took', np.round(time() - t, 2), 's .')
            print('-------------------------------------------------------------------------------------------------')
            t = time()
        # Run one test.
        cur_train_l, cur_train_ac, cur_val_l, cur_val_ac, cur_test_l, cur_test_ac = one_run()
        print(cur_val_l, cur_val_ac, cur_val_l, cur_val_ac, cur_test_l, cur_test_ac)

        # Save the current results to file, done preemptively (to not loss results) since the experiment can take long.
        train_losses.append(cur_train_l), train_accuracy.append(cur_train_ac)
        val_losses.append(cur_val_l), val_accuracy.append(cur_val_ac)
        test_losses.append(cur_test_l), test_accuracy.append(cur_test_ac)
        np.savez('cifar_10', train_losses=np.array(train_losses), train_accuracy=np.array(train_accuracy),
                 val_losses=np.array(val_losses), val_accuracy=np.array(val_accuracy),
                 test_losses=np.array(test_losses), test_accuracy=np.array(test_accuracy),
                 )


if __name__ == '__main__':
    re_run(40)
