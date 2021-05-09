import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F

from models.stylegan2.op import fused_leaky_relu, fused_leaky_relu_v


class PixelNorm(nn.Module):  # 1
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
    # rsqrt:out=1/sqrt(in)


class EqualLinear(nn.Module):  # 1
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, v=True
    ):
        super().__init__()
        '''
        nn.Parameter()
        首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
        并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，
        所以在参数优化的时候可以进行优化的)，所以经过类型转换这个self.v变成了模型的一部分，
        成为了模型中根据训练可以改动的参数了。
        使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        '''
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))  # torch中的值除以 lr_mul

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))  # 用bias_init值代替torch中的值

        else:
            self.bias = None

        self.activation = activation  # none

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.v = v

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            # t = out[0]
            # t = t[:5]
            # print('after linear', t)
            if self.v:
                out = fused_leaky_relu(out, self.bias * self.lr_mul)
                # t = out[0]
                # t = t[:5]
                # print('after leaky_relu', t)
            else:
                out = fused_leaky_relu_v(out, self.bias * self.lr_mul)
                # t = out[0]
                # t = t[:5]
                # print('after leaky_relu_v', t)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class MappingNetwork(nn.Module):
    def __init__(self, style_dim=512, n_mlp=7, lr_mlp=0.01):
        super().__init__()
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        layers.append(
            EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu',
                        v=False))  # 这里进行space W 转换space V
        self.style = nn.Sequential(*layers)

    def forward(self, styles):
        styles = [self.style(s) for s in styles]
        styles = torch.stack(styles)
        return styles


class PCA():
    def calculate_covariance_matrix(self, X, Y=None):
        # 计算协方差矩阵
        m = X.shape[0]
        X = X - np.mean(X, axis=0)
        Y = X if Y == None else Y - np.mean(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)

    def transform(self, X, n_components):
        # 设n=X.shape[1]，将n维数据降维成n_component维
        covariance_matrix = self.calculate_covariance_matrix(X)
        # 获取特征值，和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # 对特征向量排序，并取最大的前n_component组
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors[:, :n_components]
        # 转换
        return np.matmul(X, eigenvectors)


def main():
    model = MappingNetwork().cuda()

    model_dict = model.state_dict()
    ckpt = torch.load('/home/ant/pretrained_models/090000.pt')
    ckpt = ckpt['g']
    ckpt_n = {k: v for k, v in ckpt.items() if k in model_dict}
    model_dict.update(ckpt_n)
    model.load_state_dict(model_dict)

    path = '/home/ant/pixel2style2pixel-master/latent.pt'
    pt = torch.load(path)
    list = []
    i = 0
    for key in pt:
        list.append(pt[key])
        i += 1
        if i > 1000:
            break
    latent_t = torch.stack(list).cuda()  # latent_t.size(20000, 1, 512)
    latent_t_m = model(latent_t)
    latent_t_m = latent_t_m.squeeze(1).detach().cpu()
    latent_n = latent_t_m.numpy()
    # p -> p-norm
    latent_n = PCA().transform(latent_n, 500)
    latent = latent_n[:, 0]

    # latent_pd.describe()

    bin = np.arange(int(latent.min() - 0.5), int(latent.max() + 0.5), 1)
    plt.hist(latent, bins=bin, color='blue', alpha=0.5)
    plt.xlabel('value')
    plt.ylabel('num')
    plt.show()


if __name__ == "__main__":
    main()
