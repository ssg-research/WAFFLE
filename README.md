# WAFFLE: Watermarking in Federated Learning


This repo contains the Pytorch implementation of the WAFFLE: Watermarking in Federated Learning ([arXiv link](https://arxiv.org/abs/2008.07298)), and allows you to reproduce experiments presented in the paper. The paper will appear in the Proceedings of SRDS 2021.

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

### Training federated learning models

1. Train an FL model without watermarking: (Files under the folder "baseline_experiments" contains configuration files for this purpose)

`python main.py --config_file configurations/baseline_experiments/mnist/1.ini --experiment training`

2. Train an FL model with watermarking: (Files under the folder "waffle_experiments" contains configuration files for this purpose. Please check confuguration files for correctly setting the watermark set type)

`python main.py --config_file configurations/waffle_experiments/cifar10/1.ini --experiment training`

### Watermark removal attacks 

In order to run an attack (fine-tuning, pruning, Neural Cleanse), set the <code>--experiment</code> argument with the desired attack. It will run the chosen attack with different number of adversaries. 

`python main.py --config_file configurations/waffle_experiments/cifar10/1.ini --experiment fine-tuning/pruning/neural-cleanse/evasion`

You can run evasion the verification experiments by seting the <code>--experiment</code> argument to evasion. However, make sure that the tiny-imagenet dataset is downloaded into "data/datasets/tiny-imagenet-200/" folder. You can download the dataset from [here](https://www.kaggle.com/c/tiny-imagenet/overview).

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
