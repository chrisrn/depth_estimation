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
import torchvision.transforms as tt
import mediapipe as mp

from ml_pipeline.data_utils import CustomNormalization
from ml_pipeline.networks.CustomNN import ArkitCNN
from ml_pipeline.networks.Vgg import VGG
from ml_pipeline.networks.Resnets import resnet18, resnet50
from ml_pipeline.networks.EfficientNet import EfficientNet
from ml_pipeline.networks.mobilenetv2 import MobileNetV2
from ml_pipeline.networks.MoGA import MoGaA
import cv2
from facenet_pytorch import MTCNN
from torchvision.datasets.folder import has_file_allowed_extension
from mediapipe_landmarks_export import get_landmarks_image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)


def get_transforms(module):
    if module == 'arkit_blend':
        return tt.Compose([CustomNormalization(),
                           tt.Resize(size=(128, 128)),
                           tt.Grayscale(num_output_channels=1)])
    elif module == 'vgg_blend':
        return tt.Compose([CustomNormalization(),
                           tt.Resize(size=(48, 48)),
                           tt.Grayscale(num_output_channels=1)])
    elif module in ['mobilenet_blend', 'moga_blend']:
        return tt.Compose([CustomNormalization(),
                           tt.Resize(size=(224, 224)),
                           tt.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                           ])
    else:
        return tt.Compose([CustomNormalization(),
                           tt.Resize(size=(256, 256)),
                           tt.CenterCrop(224),
                           tt.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])


def get_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    return img


@ torch.no_grad()
def get_predictions(model, img, tfms, input_shape):
    img = tfms(img)
    # cv2.imshow('face', img.permute(1, 2, 0).numpy())
    img = torch.reshape(img, input_shape)
    # print(torch.std_mean(img))
    output = model(img)
    preds = torch.sigmoid(output)
    preds = torch.squeeze(preds)
    return preds


def export_predictions(output_dir, sample_name, output):
    blendshapes = []
    for i, pred in enumerate(output):
        blendshapes.append({"blendshapeName": i, "blendshapeValue": pred.item()})
    export_dict = {"faceData": blendshapes}
    with open(f'{output_dir}/{sample_name}.json', 'w') as fp:
        json.dump(export_dict, fp)


def get_model(module, device):
    if module == 'arkit_blend':
        return ArkitCNN(cnn_name='cnn_2', num_classes=52).to(device), (1, 1, 128, 128)
    elif module == 'vgg_blend':
        return VGG('VGG19', num_classes=52).to(device), (1, 1, 48, 48)
    elif module == 'efficientnet_blend':
        return EfficientNet.from_name(model_name='efficientnet-b0', num_classes=52).to(device), (1, 3, 224, 224)
    elif module == 'mobilenet_blend':
        return MobileNetV2(num_classes=52).to(device), (1, 3, 224, 224)
    elif module == 'resnet50_blend':
        return resnet50(num_classes=52).to(device), (1, 3, 224, 224)
    elif module == 'moga_blend':
        return MoGaA(n_class=52).to(device), (1, 3, 224, 224)
    else:
        return resnet18(num_classes=52).to(device), (1, 3, 224, 224)


def load_model(model_file, module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Pre-trained model loaded from: {model_file}')
    checkpoint = torch.load(model_file, map_location=device)
    # model = torch.jit.load('/Users/crountos/Desktop/work/fc_repo/face_capture/results/mobilenet_plateau_pretrained/exp_0/model/mobilenet_blend.pt')
    # input_shape = (1, 3, 224, 224)
    model, input_shape = get_model(module, device)
    new_state_dict = OrderedDict()
    for name, v in checkpoint['model_weights'].items():
        if 'module' in name:
            name = name[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.eval()

    return model, input_shape


def plot_blendshape_values(output, targets):
    plt.plot(targets, label='targets')
    plt.plot(output.detach().numpy(), label='predictions')
    plt.legend()
    plt.show()


def quality_assessment(output, targets, output_dir):
    classes = ['BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight',
               'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'EyeBlinkLeft', 'EyeBlinkRight',
               'EyeLookDownLeft', 'EyeLookDownRight', 'EyeLookInLeft', 'EyeLookInRight', 'EyeLookOutLeft',
               'EyeLookOutRight', 'EyeLookUpLeft', 'EyeLookUpRight', 'EyeSquintLeft', 'EyeSquintRight',
               'EyeWideLeft', 'EyeWideRight', 'JawForward', 'JawLeft', 'JawOpen', 'JawRight', 'MouthClose',
               'MouthDimpleLeft', 'MouthDimpleRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthFunnel',
               'MouthLeft', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthPressLeft', 'MouthPressRight',
               'MouthPucker', 'MouthRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower',
               'MouthShrugUpper', 'MouthSmileLeft', 'MouthSmileRight', 'MouthStretchLeft', 'MouthStretchRight',
               'MouthUpperUpLeft', 'MouthUpperUpRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut']
    df = pd.DataFrame(columns=['Target mean', 'Predicted mean', 'Target std', 'Predicted std', 'Mean diff', 'Std diff'],
                      index=classes)
    output = torch.stack(output)
    targets = torch.FloatTensor(targets)
    for i, label in enumerate(classes):
        preds_label = output[:, i]
        target_label = targets[:, i]
        mean_preds = torch.mean(preds_label).item()
        mean_target = torch.mean(target_label).item()
        mean_diff = abs(mean_target - mean_preds)
        std_preds = torch.std(preds_label).item()
        std_target = torch.std(target_label).item()
        std_diff = abs(std_target - std_preds)
        row = pd.DataFrame([mean_target, mean_preds, std_target, std_preds, mean_diff, std_diff])
        df.iloc[i, :] = row.T
    df = df.sort_values('Std diff')
    df.to_csv(f'{output_dir}/qa.csv')


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
    if qa:
        print(f'Perform quality assessment on images inside {input_dir}')
        quality_assessment(all_outputs, all_targets, output_dir)


def update_box(box, expand):
    return abs(int(box[0, 1])) - expand, \
           abs(int(box[0, 3])) + expand, \
           abs(int(box[0, 0])) - expand, \
           abs(int(box[0, 2])) + expand


def vanish_jitter(output, output_cleaned):
    threshold = 0.15
    count = 0
    for i in range(len(output)):
        if abs(output[i].item() - output_cleaned[i].item()) >= threshold:
            count += 1
            output_cleaned[i] = output[i]

    print(f'{count} blendshapes changed')
    return output_cleaned


def real_time_predictions(model, video_file, output_dir, tfms, mtcnn, input_shape, use_mediapipe):
    if video_file:
        cap = cv2.VideoCapture(video_file)
    else:
        # Enable web cam
        # '0' is default ID for builtin web cam
        # for external web cam ID can be 1 or -1
        cap = cv2.VideoCapture(0)

    frame_height, frame_width = 1024, 1024
    face_height, face_width = 299, 299
    height_low, height_high, width_low, width_high = 0, 299, 0, 299
    frames_per_box_update = 50
    expand_threshold = 20
    step = 0
    start_time = time.time()
    success = True

    while success:
        success, frame = cap.read()
        if success:
            frame_resized = cv2.resize(frame, (frame_height, frame_width), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            if (step == 0) or (step % frames_per_box_update == 0):
                box, _ = mtcnn.detect(frame_rgb)
                if box is None:
                    continue
                height_low, height_high, width_low, width_high = update_box(box, expand_threshold)
            face = frame_rgb[height_low:height_high, width_low:width_high]
            face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_AREA)

            # cv2.imwrite(f'{output_dir}/{step}.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{output_dir}/{step}.jpg', frame)

            if use_mediapipe:
                results = face_mesh.process(face)

                # Print and draw face mesh landmarks on the image.
                if not results.multi_face_landmarks:
                    continue
                face = get_landmarks_image(face, results)
                # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            cv2.imshow('frame', frame_resized)
            cv2.imshow('face', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            output = get_predictions(model, face, tfms, input_shape)
            if step > 0:
                output_cleaned = vanish_jitter(output, output_cleaned)
            else:
                output_cleaned = output

            export_predictions(output_dir, str(step), output_cleaned)



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
         model_file: str, module: str,
         use_mediapipe: bool, qa: bool):
    model, input_shape = load_model(model_file, module)
    mtcnn = MTCNN(image_size=299, margin=20, post_process=False)
    tfms = get_transforms(module)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    video_extensions = ('.avi', '.mov', '.mp4')
    if not input_dir or has_file_allowed_extension(input_dir, video_extensions):
        real_time_predictions(model, input_dir, output_dir, tfms, mtcnn, input_shape, use_mediapipe)
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
    parser.add_argument('--module', type=str,
                        default='arkit_blend',
                        help='Module for problem and network specification')
    parser.add_argument('--mp', action='store_true',
                        help='Use of mediapipe landmarks')
    parser.add_argument('--qa', action='store_true',
                        help='Perform quality assessment for test images')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_file, args.module, args.mp, args.qa)
