import argparse
import os
import json
import shutil
import time
import matplotlib.pyplot as plt
import statistics
from shutil import copyfile
from collections import OrderedDict
from PIL import Image
import pandas as pd
import numpy as np
import torch
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import tensorflow as tf

import cv2
from facenet_pytorch import MTCNN
from torchvision.datasets.folder import has_file_allowed_extension


def get_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    return img


def get_predictions(model, img):
    x_t = []
    image = tf.image.resize(np.expand_dims(img, axis=-1), [400, 400])
    image = np.squeeze(image, axis=-1)
    x_t.append(image)

    x_t = np.array(x_t, dtype=np.float32)
    x_t = x_t.reshape((x_t.shape[0], 400, 400, 1)).astype('float32')

    output = model(x_t)
    output = output.numpy()
    output = output[0, :53]
    output = np.delete(output, 27)
    output = output / 100
    return output


def export_predictions(output_dir, sample_name, output):
    blendshapes = []
    for i, pred in enumerate(output):
        blendshapes.append({"blendshapeName": i, "blendshapeValue": str(pred)})
    export_dict = {"faceData": blendshapes}
    with open(f'{output_dir}/{sample_name}.json', 'w') as fp:
        json.dump(export_dict, fp)


def cnn_model(num_classes):
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(400, 400, 1), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='relu'))

    return model


def load_model(model_file):
    print(f'Pre-trained model loaded from: {model_file}')
    model = cnn_model(num_classes=74)
    model.load_weights(model_file)

    return model


def plot_blendshape_values(output, targets):
    plt.plot(targets, label='targets')
    plt.plot(output.detach().numpy(), label='predictions')
    plt.legend()
    plt.show()


def files_prediction(model, input_dir, output_dir, tfms,
                     mtcnn, input_shape, qa):
    img_ext = ('.jpg', '.png')
    if input_dir.endswith('.jpg') or input_dir.endswith('.png'):
        img_files = [input_dir]
    else:
        img_files = [i for i in os.listdir(input_dir) if i.endswith(img_ext)]
    all_outputs = []
    all_targets = []
    expand_threshold = 20
    face_height, face_width = 224, 224
    for img_file in img_files:
        if input_dir.endswith('.jpg') or input_dir.endswith('.png'):
            frame_rgb = get_image(img_file)
        else:
            frame_rgb = get_image(f'{input_dir}/{img_file}')
        # box, _ = mtcnn.detect(frame_rgb)
        # if box is None:
        #     continue
        # height_low, height_high, width_low, width_high = update_box(box, expand_threshold)
        # face = frame_rgb[height_low:height_high, width_low:width_high]
        # face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_AREA)
        face = frame_rgb
        output = get_predictions(model, face, tfms, input_shape)
        sample = img_file.split('.')
        sample_name = sample[0]
        img_ext = sample[1]
        export_predictions(output_dir, sample_name, output)
        copyfile(f'{input_dir}/{sample_name}.{img_ext}', f'{output_dir}/{sample_name}.{img_ext}')
        if os.path.exists(f'{input_dir}/{sample_name}.json'):
            with open(f'{input_dir}/{sample_name}.json') as fp:
                gt = json.load(fp)
            targets = [g['blendshapeValue'] for g in gt['faceData']]
            # plot_blendshape_values(output, targets)
            all_outputs.append(output)
            all_targets.append(targets)


def update_box(box, expand):
    return abs(int(box[0, 1])) - expand, \
           abs(int(box[0, 3])) + expand, \
           abs(int(box[0, 0])) - expand, \
           abs(int(box[0, 2])) + expand


def vanish_jitter(output, output_cleaned):
    threshold = 0.15
    count = 0
    for i in range(len(output)):
        if abs(output[i] - output_cleaned[i]) >= threshold:
            count += 1
            output_cleaned[i] = output[i]

    print(f'{count} blendshapes changed')
    return output_cleaned


def real_time_predictions(model, video_file, output_dir, mtcnn):
    if video_file:
        cap = cv2.VideoCapture(video_file)
    else:
        # Enable web cam
        # '0' is default ID for builtin web cam
        # for external web cam ID can be 1 or -1
        cap = cv2.VideoCapture(0)

    frame_height, frame_width = 1077, 1616
    face_height, face_width = 400, 400
    height_low, height_high, width_low, width_high = 0, 400, 0, 400
    frames_per_box_update = 50
    expand_threshold = 150
    step = 0
    start_time = time.time()
    success = True

    while success:
        success, frame = cap.read()
        if success:
            frame_resized = cv2.resize(frame, (frame_height, frame_width), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # if (step == 0) or (step % frames_per_box_update == 0):
            #     box, _ = mtcnn.detect(frame_rgb)
            #     if box is None:
            #         continue
            #     height_low, height_high, width_low, width_high = update_box(box, expand_threshold)
            # face = frame_rgb[height_low:height_high, width_low:width_high]
            face = frame_rgb
            face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_AREA)
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

            # cv2.imwrite(f'{output_dir}/{step}.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{output_dir}/{step}.jpg', frame)

            output = get_predictions(model, face)
            if step > 0:
                output_cleaned = vanish_jitter(output, output_cleaned)
            else:
                output_cleaned = output

            export_predictions(output_dir, str(step), output_cleaned)
            cv2.imshow('frame', frame_resized)
            cv2.imshow('face', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

            step += 1
            # loop will be broken when 'q' is pressed on the keyboard
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    exec_time = time.time() - start_time
    cap.release()
    cv2.destroyWindow('face')
    print(f'{step} frames in {exec_time} seconds')
    print(f'{step / exec_time} fps')


def main(input_dir: str, output_dir: str,
         model_file: str):
    model = load_model(model_file)
    mtcnn = MTCNN(image_size=400, margin=20, post_process=False)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    video_extensions = ('.avi', '.mov', '.mp4')
    if not input_dir or has_file_allowed_extension(input_dir, video_extensions):
        real_time_predictions(model, input_dir, output_dir, mtcnn)
    else:
        files_prediction(model, input_dir, output_dir, tfms, mtcnn, input_shape, qa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='',
                        help='Input directory containing images with ground-truth json or just a video file. '
                             'If empty then real-time video predictions stored into output_dir')
    parser.add_argument('--output_dir', type=str,
                        default='../../../arkit-ml-data-extraction-unity/FaceCaptureServer/Assets/StreamingAssets/MLData',
                        help='Output directory containing images with respective json predictions')
    parser.add_argument('--model_file', type=str,
                        default='../results/arkit_all/exp_0/model/model_100.pth.tar',
                        help='Model saved on checkpoint file (.pth.tar)')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_file)
