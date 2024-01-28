import argparse
import numpy as np
import pandas as pd
import cv2 as cv

from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def image_processing(path):
    """
    Processes image
    :param path: str, path to image
    :return: ndarray, processed image
    """
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))
    # Normalization with mean-std
    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img[:, :, 0] -= mean[0]
    img[:, :, 1] -= mean[1]
    img[:, :, 2] -= mean[2]
    img[:, :, 0] /= std[0]
    img[:, :, 1] /= std[1]
    img[:, :, 2] /= std[2]
    return img


def mask_processing(path):
    """
    Processes mask
    :param path: str, path to mask
    :return: ndarray, processed mask
    """
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (224, 224))
    # Normalization with mean-std
    img = img / 255.0
    mean = (0.485 + 0.456 + 0.406) / 3.0
    std = (0.229 + 0.224 + 0.225) / 3.0
    img -= mean
    img /= std
    return img


def get_grid_params(model_name):
    """
    Returns grid parameters for model
    :param model_name: str, model name
    :return: dict, grid parameters
    """
    if model_name == 'KNN':
        return dict(n_neighbors=np.array([3, 5, 10]))
    else:
        return dict(n_estimators=np.array([3, 5, 10]))


def process_set(imgs_set, masks_set):
    """
    Processes subset of data (training or testing)
    :param imgs_set: Series, images
    :param masks_set: Series, masks
    :return: ndarrays, X, y
    """
    images, masks = [], []
    for img_path, mask_path in zip(imgs_set, masks_set):
        images.append(image_processing(img_path))
        masks.append(mask_processing(mask_path))

    images = np.stack(images)
    masks = np.stack(masks)
    X = images.reshape(-1, images.shape[-1])
    y = masks.flatten()
    return X, y


def main(data_dir, slice):
    """
    Runs the training and testing of regressor for depth estimation
    :param data_dir: str, path to data
    :param slice: int, number of samples to use
    """
    csv_file = f'{data_dir}/nyu2_train.csv'

    print('Load and process data')
    df = pd.read_csv(csv_file, header=None)
    df = '../' + df
    df = df[:slice]

    # Split the data into training and testing sets
    imgs_train, imgs_test, masks_train, masks_test = train_test_split(df[0], df[1],
                                                                      test_size=0.1,
                                                                      random_state=42)
    print(f'Num train images: {len(imgs_train)}')
    print(f'Num test images: {len(imgs_test)}')

    X_train, y_train = process_set(imgs_train, masks_train)
    X_test, y_test = process_set(imgs_test, masks_test)

    print(f'Input shape: {X_train.shape}')

    regressor = RandomForestRegressor(n_estimators=10, verbose=2)
    regressor.fit(X_train, y_train)

    print('Evaluate test set')
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    ssim = structural_similarity(y_test, y_pred, data_range=np.max(y_pred) - np.min(y_pred))
    print(f'SSIM: {ssim}')
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Dataset directory')
    parser.add_argument('--slice', type=int, default=1000,
                        help='Number of samples to slice to fit into RAM')

    args = parser.parse_args()
    main(args.data_dir, args.slice)