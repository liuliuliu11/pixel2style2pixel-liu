import torch
import pandas as pd
import matplotlib.pyplot as plt

path = '/home/ant/pixel2style2pixel-master/latent.pt'
ckpt = torch.load(path)
list = []
for key in ckpt:
    list.append(ckpt[key])
latent_t = torch.stack(list).cpu()  # latent_t.size(20000, 1, 512)
latent_t = latent_t.squeeze(1)
latent_n = latent_t.numpy()
latent = pd.DataFrame(latent_n)

index = []
for i in range(512):
    index.append(str(i + 1))

latent.columns = index
column1 = latent['1']
plt.scatter(column1.index, column1, c='r', label='feature1')  # draw the first feature.
plt.show()
