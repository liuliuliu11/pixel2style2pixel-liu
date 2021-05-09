from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils

import torchvision.transforms as transforms
import numpy as np
from scipy.io import loadmat
import cv2
import random
import torch
from datasets import augmentations

class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts

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
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), imQ]  # (0,100),higher is better,default is 95
		_, encA = cv2.imencode('.jpg', img, encode_param)
		img = cv2.imdecode(encA, 1)
		return Image.fromarray(img)

	def AddUpSample(self, img):
		return img.resize((256, 256), Image.BICUBIC)

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		# from_im = from_im.resize((256, 256))
		# from_im = transforms.ColorJitter(0.3, 0.3, 0.3, 0)(from_im)
		# from_im = self.AddUpSample(self.AddJPEG(self.AddNoise(self.AddDownSample(self.AddBlur(from_im)))))
		from_im = self.AddDownSample(from_im)

		if self.transform:
			from_im = self.transform(from_im)
		return from_im
