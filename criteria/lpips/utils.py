from collections import OrderedDict

import torch


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

'''

# origional verison

model_paths = {'alexnet': '/opt/data/private/pixel2style2pixel-master/pretrained_models/alexnet-owt-4df8aa71.pth', }

def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the`"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet_Modules(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_paths['alexnet']))  

# <class 'collections.OrderedDict'>
'''
def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    # url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v{version}/{net_type}.pth'

    # download
    # old_state_dict = torch.hub.load_state_dict_from_url(url, progress=True,map_location=None if torch.cuda.is_available() else torch.device('cpu'))
    old_state_dict = torch.load('/home/ant/pretrained_models/alex.pth')
    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
