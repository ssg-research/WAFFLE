# WAFFLE: Watermarking in Federated Learning

This paper will appear in the Proceedings of SRDS 2021.

This repo contains the Pytorch implementation of the WAFFLE: Watermarking in Federated Learning ([arXiv link](https://arxiv.org/abs/2008.07298)), and allows you to reproduce experiments presented in the paper. 

<p align="center">
  <img src="/images/Advmodel.png" width=450>
</p>

## Setup

You need to have a working conda setup (Conda 4.8.3 and Python 3.7.7 would be the best for the complete reproduction). We also assume that you have access to a GPU with CUDA >=9.2. After meeting hardware and softwarre requirements, clone this repository and install necessary software packages using conda environment.

1. Create the environment from the pysyft.yml file:
`conda env create -f pysyft.yml`
2. Activate the new environment: 
`conda activate pysyft`

We use PySyft 0.2.x, an ecosystem that supports federated learning, differential privacy and homomorphic encryption. Its codebase is in its own branch [here](https://github.com/OpenMined/PySyft/tree/syft_0.2.x), but OpenMined announced that they will not offer official support for this version range. You can visit [here](https://github.com/OpenMined/PySyft) to check the latest updates. 

## Citation
If you find our work useful in your research, you can cite the paper as follows:
```
@article{atli2020waffle,
  title={WAFFLE: Watermarking in Federated Learning},
  author={Tekgul, Buse G.A. and Xia, Yuxi and Marchal, Samuel and Asokan, N},
  eprint={2008.07298},
  archivePrefix={arXiv},
  year={2020},
  primaryClass={cs.LG}
} 
```
