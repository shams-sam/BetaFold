# w: model weight file to be loaded
# n: number of training instances used for training
# c: crop size for training
# e: number of training epochs
# d: depth of residual net
# p: dataset directory
# v: set to run only evaluation
# f: number of filters per layer
# o: output directory to save results

python run.py \
       -w ckpt/model.pt \
       -n -1 \
       -l 1e-3 \
       -c 64 \
       -e 20 \
       -b 4 \
       -d 128 \
       -p ./data/ \
       -v 0 \
       -f 64 \
       -o output/deepcon_dist_logcash/
