# Depth estimation project
## General description
In this repository 3 solutions for estimating the depth of pixels in images are implemented using the labeled [nyu_depth_v2](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2) dataset. The 1st solution is the training-evaluation of UNet architecture using PyTorch. The 2nd solution is a Random Forest Regression model using scikit learn. The 3rd solution is a ready-to-use pre-trained Transformer called DinoV2.
## Installation
Create and activate python virtual environment:
```commandline
virtualenv -p python3 venv
source venv/bin/activate
```
If your native python-version is not 3.9, you can install pyenv with the python version
you need and then:
```commandline
pyenv virtualenv 3.9.4 venv
pyenv local venv
```

Then install all the packages:
```commandline
pip install -e .
```
## How to run the UNet pipeline
There is only one script to run inside the `src` folder:
```bash
python main.py
```
which takes a default one argument, the `config.json` file. This file contains `hyper_parameters` to set up different experiments, `data` for data directory and num workers (we can use 8 if we are on GPU), `callbacks` to activate different learning rate plans and `model` for finetuning a model. Before running the experiments, you can move the `data` folder from the dataset you downloaded into the repository for better flexibility. After running, a folder named `results` contains the results of all the different experiments and we can view them on tensorboard.
The U-Net architecture is designed to capture both local and global features effectively. U-Net is originally designed for semantic segmentation tasks, where it excels at capturing fine-grained details and boundaries in images. This capability is beneficial for depth estimation, as it allows the network to differentiate between objects and understand the spatial relationships within the scene. In this solution the `segmentation_models_pytorch` library used form pytorch which provides unets with different encoders-decoders (here we use resnet18 with almost 16M parameters).
## How to run the Random Forest Regression pipeline
```bash
python main_sklearn.py
```
This script takes 2 arguments, `data_dir` for the data directory (default `../data`) and `slice` which is the number of samples to use to fit into RAM (default 1000, if we use more data the SSIM score increases but the regression runs slower). The idea behind this solution is that we use as input a 2-d array of dimensions `(num_images x image_dims, 3)` where the number of features are the 3 RGB values (after image processing). We use as labels a 1-d array of dimensions `(num_masks x mask_dims)` which are the depths to predict. The processing before training is more simple than the pytorch implementation (resize and normalization using mean and std). Random forests can handle non-linearity well, they are robust to overfitting, and they can capture complex interactions in the data. The most optimal solution is to experiment with multiple algorithms, perform cross-validation, and choose the one that performs best on each dataset.

## Evaluation
The metrics used for evaluation are the mean squared error (MSE) and the structural similarity index (SSIM). The SSIM is a perceptual metric that takes into account the structural similarity of the two images. SSIM is calculated by comparing the local patterns of the two images, taking into account the luminance, contrast, and structure of the images. SSIM is more robust to noise and small changes than MSE, but it is also more computationally expensive to calculate. In general, SSIM is preferred over MSE for image quality assessment because it provides a more accurate measure of how humans perceive the similarity of two images. However, MSE is still a useful metric, especially when speed is important.

## Ready-to-use approaches
3 state-of-the-art approaches that can be used without training are:
* [DinoV2](https://github.com/facebookresearch/dinov2) transformer from facebook research. This is also a solution which can be used by running the `dinov2` notebook which loads the pre-trained model from [Hugging Face](https://huggingface.co/facebook/dpt-dinov2-base-nyu) and you can see how to run it inside the notebook.
* [EVP](https://github.com/lavreniuk/evp) (Enhanced Visual Perception) which exploits also text content for the depth estimation.
* [MIM](https://github.com/SwinTransformer/MIM-Depth-Estimation) (Masked Image Modeling) which provides pre-trained transformers on the NYU dataset.