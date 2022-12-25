# BetaFold

## An attempt to recreate AlphaFold using PyTorch for regular researchers

## Setup

```bash
conda create --name betafold python=3.8
pip install -r requirements.txt
```

## Download dataset

```bash
wget http://deep.cs.umsl.edu/pdnet/train-data.tar.gz
tar zxvf train-data.tar.gz
rm train-data.tar.gz
mkdir output
```

## Training

```bash
sh sh/train.sh dist_logcosh
```

## Evaluation

```bash
sh sh/train.sh dist_logcosh
```

## References

- [1] Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., ˇZ ́ıdek, A., Potapenko, A., et al. Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873):583–589, 2021.
- [2] Senior, A. W., Evans, R., Jumper, J., Kirkpatrick, J., Sifre, L., Green, T., Qin, C., ˇZ ́ıdek, A., Nelson, A. W., Bridgland, A., et al. Improved protein structure prediction using potentials from deep learning. Nature, 577(7792):706–710, 2020.
- [2] Adhikari, B. A fully open-source framework for deep learning protein real-valued distances. Scientific reports, 10(1):1–10, 2020.

