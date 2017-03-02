python ./GeneralCLS/train.py --gpu 1  \
                --solver models/GeneralCLS/solver.prototxt \
                --weights pretrained_models/bvlc_googlenet.caffemodel \
                --iters 800000 \
                --size 224 \
                --imdb GeneralCLS \
                --out out \
                --batchsize 2 
