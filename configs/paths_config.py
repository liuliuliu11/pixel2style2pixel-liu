dataset_paths = {
	'celeba_train': '/home/ant/pixel2spDatasets/FalseDatasets/train_img',
	'celeba_test': '/home/ant/pixel2spDatasets/FalseDatasets/valid_img',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '/home/ant/pixel2spDatasets/FalseDatasets/train_img',
}

model_paths = {
	'stylegan_ffhq': '/home/ant/pretrained_models/090000.pt',  # leye
	# 'stylegan_ffhq': '/home/ant/pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': '/home/ant/pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat'
}
