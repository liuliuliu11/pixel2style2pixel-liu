from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, default='/home/ant/path_result_true_1020', help='Path to experiment output directory')
		self.parser.add_argument('--latent_dir', type=str, default='/home/ant/zi-latent-true', help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default='/home/ant/pixel2style2pixel-master/path/to/experiment/checkpoints/best_model.pt', type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--data_path', type=str, default='/home/ant/ffhq_about/FFHQ1024Few-all/1000', help='Path to directory of images to evaluate')
		self.parser.add_argument('--couple_outputs', action='store_true', help='Whether to also save inputs + outputs side-by-side')

		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		# arguments for style-mixing script
		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')
		self.parser.add_argument('--n_outputs_to_generate', type=int, default=5, help='Number of outputs to generate per input image.')
		self.parser.add_argument('--mix_alpha', type=float, default=None, help='Alpha value for style-mixing')
		self.parser.add_argument('--latent_mask', type=str, default=None, help='Comma-separated list of latents to perform style-mixing with')

		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default='1', help='Downsampling factor for super-res (should be a single value for inference).')

	def parse(self):
		opts = self.parser.parse_args()
		opts.couple_outputs = True
		return opts