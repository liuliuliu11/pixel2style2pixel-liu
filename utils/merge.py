import numpy as np
import torch

'''

latent_dir = '/home/ant/zi-latent-true'
feature = []
for i in range(0, 1000, 2):
    path = latent_dir + '/latent_{}.npz'.format(i)
    enc = np.load(path)
    nlatent = enc['arr_0']  # nlatent=<class tuple>:(2,14,512)
    for l in nlatent:  # l=<class tuple>:(14,512)
        t = torch.from_numpy(l[0])
        print(t.size)
        feature.append(t)  # l[0]=<class 'numpy.ndarray'>:512

ff = torch.stack(feature)
ff = ff.cpu().numpy()
np.savez('/home/ant/Truelatent', ff)


'''
import pandas as pd
import matplotlib.pyplot as plt

enc = np.load('/home/ant/Truelatent.npz')
latent = enc['arr_0']
latent = pd.DataFrame(latent)

index = []
for i in range(512):
    index.append(str(i + 1))

latent.columns = index
column1 = latent['1']
bin = np.arange(int(column1.min() - 0.5), int(column1.max() + 0.5), 0.05)
plt.hist(column1, bins=bin, color='blue', alpha=0.5)
plt.xlabel('value')
plt.ylabel('num')
plt.show()



