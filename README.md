# Smart init: Deep learning as (PAC) learning.
The official code of the paper **Smart init: Deep learning as (PAC) learning**,
which suggested an initialization method that have improved the results in the experiments done in the paper.

This repository contains the code and results for all the algorithms and examples that appeared in the paper.
### CONSTRUCTION
- **paper** the current version of the paper.
- **results** contains all the results for the tests.
- **data** contains datasets used, and their extracted version.
- **cifar** contains the tests done for cifar-10, to change between the tests modify directly.
- **data_extrac** code to extract imagenet from *tensorflow_datasets* and transform it to torch tensor. 
This allows easier integration with PyTorch.
- **Examples** the code used to generate Figure 1 in the paper, i.e., the gradient decent motivation.
- **image_augment** our try to augment the images, was not used since this seemed to worsen the results.
Used to apply `torch.transpose` on the images.
- **main** the main code of the paper that was used to run the main experiments over imagenet 
(the code for Figure 5 is in over_train), to change between the tests modify directly.
- **Mobile** generates the MobileNet pytorch object, taken from https://github.com/jmjeon94/MobileNet-Pytorch.
- **over_train** generates the results for Figure 5 in the paper.
- **plot** the code that generates the plots (besides Figure 5) and the values in the tables in the paper.


