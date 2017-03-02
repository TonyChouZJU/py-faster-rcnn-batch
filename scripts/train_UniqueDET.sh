python ./UniqueDET/train.py --gpu 0  \
                --solver models/UniqueDET/solver.prototxt \
                --weights pretrained_models/VGG_CNN_M_1024.v2.caffemodel \
                --iters 100000 \
                --size 224 \
                --imdb UniqueDET \
                --out out \
                --batchsize 1 \
                --cfg experiments/cfgs/faster_rcnn_end2end.yml
