# w: model weight file to be loaded
# n: number of training instances used for training
# c: crop size for training
# e: number of training epochs
# d: depth of residual net
# p: dataset directory
# v: set to run only evaluation
# f: number of filters per layer
# o: output directory to save results

# deepcon_dist_logcash: lr=1e-2
# deepcon_dist_logcash_v1: lr=1e-3
# deepcon_dist_logcash_v2: lr=1e-2 feat-eng=100/d
# deepcon_dist_logcash_v3: lr=1e-3 feat-eng=100/d
dist_logcosh(){
    python run.py -w ckpt/model.pt -n -1 -c 64 \
       -l 1e-3 -e 10 -b 8 -v 0 -i 0 -t dist\
       -d 128 -p ./data/ -f 64 \
       -o output/deepcon_dist_logcash_v3/
}

# deepcon_dist_invlogcash: lr=1e-2 time=1hr30min
# deepcon_dist_invlogcash_v1: lr=1e-3 time=1hr33min
# deepcon_dist_invlogcash_v2: lr=1e-2 epochs=100 time=6hr46min
# deepcon_dist_invlogcash_v3: lr=1e-3 epochs=100
dist_invlogcosh(){
    python run.py -w ckpt/model.pt -n -1 -c 64 \
       -l 1e-3 -e 100 -b 8 -v 0 -i 1 -t dist\
       -d 128 -p ./data/ -f 64 \
       -o output/deepcon_dist_invlogcash_v3/
}

contact(){
    python run.py -w ckpt/model.pt -n -1 -c 64 \
       -l 1e-2 -e 20 -b 8 -v 0 -i 1 -t contact\
       -d 128 -p ./data/ -f 64 \
       -o output/deepcon_contact_v1/
}

binned(){
    python run.py -w ckpt/model.pt -n -1 -c 64 \
       -l 1e-2 -e 20 -b 8 -v 0 -i 1 -t binned\
       -d 128 -p ./data/ -f 64 \
       -o output/deepcon_binned/
}

$1
