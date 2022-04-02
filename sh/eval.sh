dist_invlogcosh(){    
    python run.py -w ckpt/model.pt -n -1 -c 64 \
       -l 1e-3 -e 100 -b 8 -v 1 -i 1 -t dist\
       -d 128 -p ./data/ -f 64 \
       -o output/deepcon_dist_invlogcash_v2/

    python run.py -w ckpt/model.pt -n -1 -c 64 \
       -l 1e-3 -e 100 -b 8 -v 1 -i 1 -t dist\
       -d 128 -p ./data/ -f 64 \
       -o output/deepcon_dist_invlogcash_v3/
}

dist_logcosh(){
    python run.py -w ckpt/model.pt -n -1 -c 64 \
       -l 1e-3 -e 10 -b 8 -v 1 -i 0 -t dist\
       -d 128 -p ./data/ -f 64 \
       -o output/deepcon_dist_logcash_v2/

    python run.py -w ckpt/model.pt -n -1 -c 64 \
       -l 1e-3 -e 10 -b 8 -v 1 -i 0 -t dist\
       -d 128 -p ./data/ -f 64 \
       -o output/deepcon_dist_logcash_v3/
}

contact(){
    python run.py -w ckpt/model.pt -n -1 -c 64 \
       -l 1e-2 -e 20 -b 8 -v 1 -i 1 -t contact\
       -d 128 -p ./data/ -f 64 \
       -o output/deepcon_contact_v1/
}


$1
