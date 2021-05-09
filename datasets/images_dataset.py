from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils

import face_alignment
from skimage import io
import torchvision.transforms as transforms
import numpy as np
import os
from scipy.io import loadmat
import cv2
import random
import torch
import time
from functools import wraps
from datasets import augmentations


def loop_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(600):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError:
                time.sleep(1)
        return ret

    return wrapper


class ImagesDataset(Dataset):

    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):

        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform  # downscale
        self.target_transform = target_transform  # constant
        self.opts = opts

        # Initialize alignment
        # self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0', flip_input=False)

    def AddNoise(self, img):  # noise
        if random.random() > 0.9:  #
            return img
        self.sigma = np.random.randint(1, 11)
        img_tensor = torch.from_numpy(np.array(img)).float()
        noise = torch.randn(img_tensor.size()).mul_(self.sigma / 1.0)

        noiseimg = torch.clamp(noise + img_tensor, 0, 255)
        return Image.fromarray(np.uint8(noiseimg.numpy()))

    def AddBlur(self, img):  # gaussian blur or motion blur
        if random.random() > 0.9:  #
            return img
        img = np.array(img)
        if random.random() > 0.35:  ##gaussian blur
            blursize = random.randint(1, 17) * 2 + 1  ##3,5,7,9,11,13,15
            blursigma = random.randint(3, 20)
            img = cv2.GaussianBlur(img, (blursize, blursize), blursigma / 10)
        else:  # motion blur
            M = random.randint(1, 32)
            KName = '/home/ant/pixel2style2pixel-master/datasets/MotionBlurKernel/m_%02d.mat' % M
            k = loadmat(KName)['kernel']
            k = k.astype(np.float32)
            k /= np.sum(k)
            img = cv2.filter2D(img, -1, k)
        return Image.fromarray(img)

    def AddDownSample(self, img):  # downsampling
        if self.opts.resize_factors is None:
            self.opts.resize_factors = '1,2,4,8,16,32'
        factors = [int(f) for f in self.opts.resize_factors.split(",")]
        # print("Performing down-sampling with factors: {}".format(factors))
        down = augmentations.BilinearResize(factors=factors)
        return down(img)

    def AddJPEG(self, img):  # JPEG compression
        if random.random() > 0.6:  #
            return img
        imQ = random.randint(40, 80)
        img = np.array(img)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), imQ]  # (1,100),higher is better,default is 95
        _, encA = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encA, 1)
        return Image.fromarray(img)

    def AddUpSample(self, img):
        return img.resize((self.opts.fineSize, self.opts.fineSize), Image.BICUBIC)

    def __len__(self):
        return len(self.source_paths)

    @loop_until_success
    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

        from_im = from_im.resize((self.opts.fineSize, self.opts.fineSize))
        from_im = transforms.ColorJitter(0.3, 0.3, 0.3, 0)(from_im)
        from_im = self.AddUpSample(self.AddJPEG(self.AddNoise(self.AddDownSample(self.AddBlur(from_im)))))

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert('RGB')
        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im

        return from_im, to_im  # in ffhq-encode's task from_im = to_im (source_im is after processing)


'''
 def get_part_location(self, imgname):
        # 5leyetrain/image/61724.png
        # ioimg = io.imread(path)
        # PredsAll = []
        # try:
        #     PredsAll = self.fa.get_landmarks(ioimg)
        # except:
        #     print('#########No face')
        # if PredsAll is None:
        #     print('#########No face2')
        # if len(PredsAll) != 1:
        #     print('#########too many face')
        # Landmarks = PredsAll[-1]

        ##### noted by yangliu
        
        # here using fa to input the face's image to obtain the landmarks is a failure(error:No face---No face2), so the data is input by reading the landmarks file.

        

        Landmarks = []
        if imgname[-21:-16] == 'train':
            landmarkpath = '/home/ant/ffhq_about/512-face-afteralign/FFHQ512Landmarksaligntrain'
        elif imgname[-20:-16] == 'test':
            landmarkpath = '/home/ant/ffhq_about/512-face-afteralign/FFHQ512Landmarksaligntest'
        else:
            landmarkpath = '/home/ant/ffhq_about/512-face-afteralign/FFHQ512Landmarksalignvalid'
        with open(os.path.join(landmarkpath, imgname[-9:] + '.txt'), 'r') as f:
            for line in f:
                tmp = [np.float(i) for i in line.split(' ') if i != '\n']
                Landmarks.append(tmp)
        Landmarks = np.array(Landmarks)
        # landmark.shape #(68,2)  type(landmark) #<class 'numpy.ndarray'>

        # modified by yangliu in line133-line155
        Map_LE = list(np.hstack((range(17, 22), range(36, 42))))
        Map_RE = list(np.hstack((range(22, 27), range(42, 48))))
        Map_NO = list(range(29, 36))
        Map_MO = list(range(48, 68))
        # left eye
        Mean_LE = np.mean(Landmarks[Map_LE], 0)
        L_LE = np.max((np.max(np.max(Landmarks[Map_LE], 0) - np.min(Landmarks[Map_LE], 0)) / 2, 16))
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        # right eye
        Mean_RE = np.mean(Landmarks[Map_RE], 0)
        L_RE = np.max((np.max(np.max(Landmarks[Map_RE], 0) - np.min(Landmarks[Map_RE], 0)) / 2, 16))
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        # nose
        Mean_NO = np.mean(Landmarks[Map_NO], 0)
        L_NO = np.max((np.max(np.max(Landmarks[Map_NO], 0) - np.min(Landmarks[Map_NO], 0)) / 2, 16))
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        # mouth
        Mean_MO = np.mean(Landmarks[Map_MO], 0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO], 0) - np.min(Landmarks[Map_MO], 0)) / 2, 16))
        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)

        return Location_LE, Location_RE, Location_NO, Location_MO
'''
