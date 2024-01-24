import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
from sklearn.model_selection import train_test_split


class DepthDataset:
    def __init__(self, df, tfms):
        self.df = df
        self.tfms = tfms

    def open_im(self, p, gray=False):
        im = cv.imread(str(p))
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY if gray else cv.COLOR_BGR2RGB)
        return im

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        s = self.df.iloc[idx, :]
        im, dp = s[0], s[1]
        im, dp = self.open_im(im), self.open_im(dp, True)
        augs = self.tfms(image=im, mask=dp)
        im, dp = augs['image'], augs['mask'] / 255.
        return im, dp.unsqueeze(0)


class DepthDataHandler:
    def __init__(self, data_params: dict, batch_size: int):
        self.data_dir = data_params['data_dir']
        self.num_workers = data_params['num_workers']
        self.batch_size = batch_size

    def get_transforms(self):
        sample_tfms = [
            A.HorizontalFlip(),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.5),
            ], p=0.3),
            A.RGBShift(),
            A.RandomBrightnessContrast(),
            A.RandomResizedCrop(384, 384),
            A.ColorJitter(),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
            A.HueSaturationValue(p=0.3),
        ]
        train_tfms = A.Compose([
            *sample_tfms,
            A.Resize(224, 224),
            A.Normalize(always_apply=True),
            ToTensorV2()
        ])

        valid_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(always_apply=True),
            ToTensorV2()
        ])
        return train_tfms, valid_tfms

    def get_data(self):
        csv_file = f'{self.data_dir}/nyu2_train.csv'

        df = pd.read_csv(csv_file, header=None)
        df[0] = df[0].map(lambda x: f'../{x}')
        df[1] = df[1].map(lambda x: f'../{x}')

        train_df, val_df = train_test_split(df, test_size=0.1, shuffle=True)
        val_df, test_df = train_test_split(val_df, test_size=0.1, shuffle=True)
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        print(f'Num train samples: {len(train_df)}')
        print(f'Num val samples: {len(val_df)}')
        print(f'Num test samples: {len(test_df)}')

        train_tfms, valid_tfms = self.get_transforms()
        train_ds = DepthDataset(train_df, train_tfms)
        val_ds = DepthDataset(val_df, valid_tfms)
        test_ds = DepthDataset(test_df, valid_tfms)

        # PyTorch data loaders
        train_loader = DataLoader(train_ds, self.batch_size, shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_ds, self.batch_size, shuffle=True,
                                num_workers=self.num_workers,
                                pin_memory=True)
        test_loader = DataLoader(test_ds, self.batch_size, shuffle=True,
                                 num_workers=self.num_workers,
                                 pin_memory=True)


        return train_loader, val_loader, test_loader
