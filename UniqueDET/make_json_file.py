import json

basic_priority = 0

data = {
	'info':{'time':'2016.11.11',
		'team':'wl_algorithm_dl',
		'task':'union classification and detection'},
	'params':[
		{'dl_type':0,
		 'dl_name':'GEN_CLS',
		 'priority':basic_priority,
		 'test_prototxt':r'/home/tomorning/Workspace/Recognition/models/GeneralCLS/test_fc.prototxt',
		 'net_model':r'/home/tomorning/Workspace/Recognition/out/googlenet_home_GCLS_iter_800000.caffemodel',
		 'mean_file':r'/home/tomorning/Workspace/Recognition/data/GeneralCLS/mean.npy',
		 'sysnets':r'/home/tomorning/Workspace/Recognition/data/GeneralCLS/sysnets.txt'
		},

		{'dl_type':1,
		 'dl_name':'GEN_DET',
		 'priority':basic_priority+1,
		 'test_prototxt':r'/home/tomorning/Workspace/Recognition/models/GeneralDET/coco/VGG16/faster_rcnn_end2end/test.prototxt',
		 'net_model':r'/home/tomorning/Workspace/Recognition/out/faster_rcnn_end2end/coco/coco_vgg16_faster_rcnn_final.caffemodel',
		 'mean_file':None,
		 'sysnets':'/home/tomorning/Workspace/Recognition/data/GeneralDET/coco_sysnets.txt',
		 'next':
			{
		 	  	'dl_name':'DLT_CLS',
		          	'priority':basic_priority+1,
		 	  	'test_prototxt':r'/home/tomorning/Workspace/Recognition/models/DetailCLS/car/googlenet/test.prototxt',
		 		'net_model':r'/home/tomorning/Workspace/Recognition/out/cls/detail_car/googlenet_finetune_web_car_iter_10000.caffemodel',
		 		'mean_file':r'/home/tomorning/Workspace/Recognintion/data/DetailCLS/imagenet_mean256.npy',
		 		'sysnets':r'/home/tomorning/Workspace/Recognition/data/DetailCLS/car_sysnets.txt'
			}
		},

		{'dl_type':1,
		 'dl_name':'UNQ_DET',
		 'priority':basic_priority+1,
		 'test_prototxt':r'/home/tomorning/Workspace/Recognition/models/UniqueDET/brands_144_det/VGG_CNN_M_1024/faster_rcnn_end2end/test.prototxt',
		 'net_model':r'/home/tomorning/Workspace/Recognition/out/faster_rcnn_end2end/brands_144/brands_vgg_cnn_m_1024_faster_rcnn_iter_80000.caffemodel',
		 'mean_file':None,
		 'sysnets':'/home/tomorning/Workspace/Recognition/data/UniqueDET/brands_144_sysnets.txt'
		},

		{'dl_type':1,
		 'dl_name':'UNQ_DET',
		 'priority':basic_priority+1,
		 'test_prototxt':r'/home/tomorning/Workspace/Recognition/models/UniqueDET/item_logo_15_det/VGG16/faster_rcnn_end2end/test.prototxt',
		 'net_model':r'/home/tomorning/Workspace/Recognition/out/faster_rcnn_end2end/item_logo_15/UDET_vgg_cnn_m_1024_iter_10000.caffemodel',
		 'mean_file':None,
		 'sysnet':'/home/tomorning/Workspace/Recognition/data/UniqueDET/item_logo_15_sysnets.txt'
		}
		
			
		]



}

with open('./union_test.json','w+') as f:
	f.write(json.dumps(data))
