import torch
import os
import numpy as np
from typing import List, Tuple, Any, Dict
import torchvision as tv
import torchvision.models as models
import configparser
import data_handle as dh
import torch.nn as nn
from collections import namedtuple
import torch.nn.functional as F
import torch.optim as optim
import Network
import logger
import random
import syft as sy

log = logger.Logger(prefix=">>>")



def prepare_data(config: configparser.ConfigParser) -> namedtuple:
    model_save_path = config["DEFAULT"]["model_save_path"]
    if not os.path.exists(model_save_path):
        log.warn(model_save_path + " does not exist. Creating...")
        os.mkdir(model_save_path)
        log.info(model_save_path + " Created.")

    scores_save_path = config["DEFAULT"]["scores_save_path"]
    if not os.path.exists(scores_save_path):
        log.warn(scores_save_path + " does not exist. Creating...")
        os.mkdir(scores_save_path)
        log.info(scores_save_path + " Created.")

    # DOWNLOAD DATASETS
    train_dataset = config["DEFAULT"]["dataset_name"]
    train_dataset_save_path = config["FEDERATED"]["train_dataset_save_path"]
    watermark_data_save_path = config["WATERMARK"]["watermark_dataset_save_path"]
    watermark_model_save_path = config["WATERMARK"]["watermark_model_save_path"]
    watermark_dataset = config["WATERMARK"]["watermark_set"]
    train_batch_size = int(config["DEFAULT"]["train_batch_size"])
    test_batch_size = int(config["DEFAULT"]["test_batch_size"])
    number_of_classes = int(config["DEFAULT"]["number_of_classes"])
    force_greyscale = config["WATERMARK"].getboolean("force_greyscale")
    normalize_with_imagenet_vals = config["WATERMARK"].getboolean("normalize_with_imagenet_vals")
    client_num = int(config["DEFAULT"]["client_num"])
    input_size = int(config["DEFAULT"]["input_size"])
    pretrain_epochs = int(config["WATERMARK"]["pretrain_epochs"])
    pretrain = int(config["WATERMARK"]["pretrain"])
    w_retrain = int(config["WATERMARK"]["w_retrain"])
    retrain = int(config["FEDERATED"]["retrain"])
    model_architecture = config["FEDERATED"]["model_architecture"]
    epochs = int(config["DEFAULT"]["epochs"])




    training_transforms, watermark_transforms = setup_transformations(train_dataset, watermark_dataset,
                                                                      force_greyscale, normalize_with_imagenet_vals,
                                                                      input_size)
    # DOWNLOAD TRAIN DATASET
    train_set, test_set = download_traindata(train_dataset, train_dataset_save_path, training_transforms)

    # DOWNLOAD WATERMARK DATASET
    watermark_size = int(config["WATERMARK"]["watermark_size"])



    # SUBCLASS TRAINING SET IF THE SETS ARE THE SAME, OTHERWISE JUST TAKE SAMPLES
    watermark_set = download_watermark(watermark_dataset, watermark_data_save_path, watermark_transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)

    # client, federated_train_loader = setup_federated_trainloader(train_loader, client_num, train_batch_size)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size = test_batch_size)


    watermark_loader = torch.utils.data.DataLoader(watermark_set, batch_size=50, shuffle = True)

    # SETUP TRAINING MODEL
    federated_retrain = config["FEDERATED"].getboolean("retrain")
    attacker_model_path = "./data/attacker_model/" + config["WATERMARK"]["model_name"]
    if pretrain or w_retrain or retrain:
        federated_model_path = watermark_model_save_path + config["WATERMARK"]["model_name"]
    else:
        federated_model_path = model_save_path + config["FEDERATED"]["model_name"]
    federated_model = setup_model(
        retrain=federated_retrain,
        model_architecture=model_architecture,
        model_path=federated_model_path,
        number_of_classes = number_of_classes
    )

    # SETUP WATERMARK MODEL
    watermark_retrain = config["WATERMARK"].getboolean("retrain")

    watermark_model = setup_model(
        retrain=watermark_retrain,
        model_architecture=model_architecture,
        model_path=watermark_model_save_path,
        number_of_classes=number_of_classes
    )

    # SETUP TRAINING PROCEDURE
    TrainingOps = namedtuple("TrainingOps",
                             [
                                 "epochs",
                                 "local_epoch",
                                 "client_num",
                                 "federated_model",
                                 "model_architecture",

                                 "fl_model_name",
                                 "use_cuda",
                                 "train_set",
                                 "resume_from_checkpoint_path",
                                 "checkpoint_save_path"

                             ])

    training_ops = TrainingOps(
        epochs,
        int(config["DEFAULT"]["local_epoch"]),
        int(config["DEFAULT"]["client_num"]),
        federated_model,
        model_architecture,
        config["FEDERATED"]["model_name"],
        config["DEFAULT"]["use_cuda"],
        train_set,
        int(config["FEDERATED"]["resume"]),
        config["DEFAULT"]["checkpoint_save_path"],

    )

    # SETUP TEST PROCEDURE
    TestOps = namedtuple("TestOps", [
                                        "test_loader",
                                        "test_set",
                                        "use_cuda",
                                        "test_batch_size",
                                        "number_of_classes"
                                    ])
    test_ops = TestOps(
        test_loader,
        test_set,
        config["DEFAULT"]["use_cuda"],
        test_batch_size,
        number_of_classes
    )

    # SETUP WATERMARK EMBEDDING
    WatermarkOps = namedtuple("WatermarkOps",
                              [
                                  "epochs",
                                  "watermark_model",
                                  "use_cuda",
                                  "training_loader",
                                  "watermark_loader",
                                  "number_of_classes",
                                  "weight_decay",
                                  "learning_rate",
                                  "watermark_data_save_path",
                                  "watermark_model_save_path",
                                  "watermark_transforms",
                                  "pretrain",
                                  "pretrain_epochs",
                                  "w_retrain",
                                  "model_name"
                              ])

    watermark_ops = WatermarkOps(
        epochs,
        watermark_model,
        config["DEFAULT"]["use_cuda"],
        train_loader,
        watermark_loader,
        number_of_classes,
        float(config["WATERMARK"]["decay"]),
        float(config["WATERMARK"]["learning_rate"]),
        watermark_data_save_path,
        watermark_model_save_path,
        watermark_transforms,
        pretrain,
        pretrain_epochs,
        w_retrain,
        config["WATERMARK"]["model_name"]
    )

    # SETUP EXPERIMENT ENVIRONMENT
    Environment = namedtuple("Environment",
                             [
                                 "federated_retrain",
                                 "watermark_retrain",
                                 "federated_model_path",
                                 "watermark_model_path",
                                 "training_ops",
                                 "test_ops",
                                 "watermark_ops",
                                 "attack_model_path",
                                 "watermark_model_save_path"
                             ])
    return Environment(
        federated_retrain,
        watermark_retrain,
        federated_model_path,
        watermark_model_save_path,
        training_ops,
        test_ops,
        watermark_ops,
        attacker_model_path,
        watermark_model_save_path
    )


def setup_transformations(training_set: str, watermark_set: str, force_greyscale: bool, normalize_with_imagenet_vals: bool, input_size: int):
    if training_set == "MNIST":
        mean = [0.5]#[0.1307]
        std = [0.5]#[0.3081]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    if normalize_with_imagenet_vals:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    #if training_set == "CINIC10":
    #    mean = [0.47889522, 0.47227842, 0.43047404]
    #    std = [0.24205776, 0.23828046, 0.25874835]

    train_transforms = {
        "MNIST": {
            "train": tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
            "val": tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ])
        },
        "CIFAR10": {
            'train': tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
            'val': tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
        },
        "CINIC10":{
            'train': tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
            'val': tv.transforms.Compose([
                tv.transforms.Resize(input_size),
                tv.transforms.CenterCrop(input_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std)
            ]),
        }

    }

    greyscale = [tv.transforms.Grayscale()] if force_greyscale else []
    watermark_transforms = {
        "MPATTERN": tv.transforms.Compose(greyscale + [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ]),
        "CPATTERN": tv.transforms.Compose(greyscale + [
            tv.transforms.Resize(input_size),
            tv.transforms.CenterCrop(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ]),
        "IMAGENET": tv.transforms.Compose(greyscale + [
            tv.transforms.Resize(input_size),
            tv.transforms.CenterCrop(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ]),
        "PATTERN":tv.transforms.Compose(greyscale + [
            tv.transforms.Resize(input_size),
            tv.transforms.CenterCrop(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ]),
        "RANDOM":tv.transforms.Compose(greyscale + [
            tv.transforms.Resize(input_size),
            tv.transforms.CenterCrop(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean, std)
        ]),
    }

    train_transform = train_transforms[training_set]
    if train_transform is None:
        log.error("Specified training set transform is not available.")
        raise ValueError(training_set)

    watermark_transform = watermark_transforms[watermark_set]
    if watermark_transform is None:
        log.error("Specified watermark set transform is not available.")
        raise ValueError(watermark_set)

    return train_transform, watermark_transform

#
# transform = transforms.Compose(
#     [
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

def download_traindata(dataset_name: str, data_path: str, transformations: Dict[str, tv.transforms.Compose]):
    if dataset_name == "MNIST":
        dataset = tv.datasets.MNIST
    elif dataset_name == "CIFAR10":
        dataset = tv.datasets.CIFAR10
    elif dataset_name == "CINIC10":
        dataset = dh.CINIC10
    else:
        raise ValueError(dataset_name)
    if dataset_name == "MNIST" or dataset_name == "CIFAR10":
        train_set = dataset(data_path, train=True, transform=transformations["train"], download=True)
        test_set = dataset(data_path, train=False, transform=transformations["val"], download=True)
    elif dataset_name == "CINIC10":
        train_set = tv.datasets.ImageFolder(data_path + '/train',transform=transformations["train"])
        test_set = tv.datasets.ImageFolder(data_path + '/train',transform=transformations["val"])

    log.info("Training ({}) samples: {}\nTest samples: {}\nSaved in: {}".format(dataset_name, len(train_set), len(test_set), data_path))
    return train_set, test_set


def get_with_default(config: configparser.ConfigParser, section: str, name: str, type_, default=None):
    if config.has_option(section, name):
        return type_(config.get(section, name))
    else:
        return default


def setup_model(retrain: bool, model_architecture: str, model_path: str, number_of_classes: int) -> nn.Module:
    available_models = {
        "MNIST_L5": Network.MNIST_L5,
        "vgg16": Network.vgg16
    }

    #if model_architecture == "vgg16":
    #    model = available_models[model_architecture](pretrained=True)
    #    n_features = model.classifier[6].in_features
    #    model.classifier[6] = nn.Linear(n_features, number_of_classes)
    #else:
    #    model = available_models[model_architecture]()
    model = available_models[model_architecture]
    if model_architecture == "vgg16":
        n_features = model().classifier[6].in_features
        model().classifier[6] = nn.Linear(n_features, number_of_classes)
    if model is None:
        log.error("Incorrect model architecture specified or architecture not available.")
        raise ValueError(model_architecture)

    return model


def download_watermark(watermark_dataset_name: str, watermark_data_path: str, transformations: tv.transforms.Compose):
    if watermark_dataset_name == "CPATTERN":
        watermark_set = dh.ImageNet(watermark_data_path, train=False, transform=transformations, download=True)
    elif watermark_dataset_name == "PATTERN" or "RANDOM" or "IMAGENET" or "MPATTERN":
        watermark_set = dh.Pattern(watermark_data_path, train=False, transform=transformations, download=True)
    else:
        raise ValueError(watermark_dataset_name)

    log.info("Watermark ({}) samples: {}\nSaved in: {}".format(watermark_dataset_name, len(watermark_set), watermark_data_path))

    return watermark_set



