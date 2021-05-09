"""
This file runs the main training/val loop

python scripts/train.py --dataset_type=celebs_super_resolution --exp_dir=/home/ant/pixel2style2pixel-master/path/to/experiment /
--workers=2 --batch_size=2 --test_batch_size=2 --test_workers=2 --val_interval=2500 --save_interval=5000 /
--encoder_type=GradualStyleEncoder --start_from_latent_avg /
--lpips_lambda=0.8 --l2_lambda=1 --id_lambda=0.1 --w_norm_lambda=0 --resize_factors=1,2,4,8,16,32 --max_steps=40000


id_loss: model_ir_se50.pth
lLPIPS_loss: alex-owt-4df8aa71.pth & alex.pth

"""
import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach

def main():
    opts = TrainOptions().parse()
    if not os.path.exists(opts.exp_dir):
        os.makedirs(opts.exp_dir)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    coach = Coach(opts)
    coach.train()


if __name__ == '__main__':
    main()
