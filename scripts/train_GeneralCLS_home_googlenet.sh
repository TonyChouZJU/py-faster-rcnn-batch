python ./GeneralCLS/train.py --gpu 1  \
                --solver models/GeneralCLS/googlenet_solver.prototxt \
                --weights out/googlenet_home_GCLS_iter_800000.caffemodel \
                --iters 800000 \
                --size 224 \
                --imdb GeneralCLS \
                --out out \
                --batchsize 1 
