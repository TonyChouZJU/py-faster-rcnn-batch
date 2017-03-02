python ./GeneralCLS/train.py --gpu 1  \
                --solver models/GeneralCLS/solver.prototxt \
                --weights out/home_GCLS_iter_700000.caffemodel \
                --iters 800000 \
                --size 224 \
                --imdb GeneralCLS \
                --out out \
                --batchsize 1
