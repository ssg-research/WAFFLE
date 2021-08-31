import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from collections import namedtuple
from collections import defaultdict
from typing import Dict, Any, List
import os,sys
from scipy import stats
import numpy as np
import score
# KEYS ARE:
#   - VICTIM: test_average, test_per_class, loss
#   - ATTACKER: test_averate, test_per_class, test_watermark, loss


def load_file(file_path: str) -> Dict[str, Any]:

    with open(file_path, "rb") as f:
        return pickle.load(f)


def plot_training(files: List[str]) -> None:
    # sns.set()
    scores = []
    test_average = []
    loss = []
    last_tick = []
    number_of_observations = []
    epochs = []
    step = 5
    for i in range(5):
        if files[i] is None:
            scores.append(0)
            test_average.append(0)
            number_of_observations.append(0)
            epochs.append(0)
        else:
            print(i)
            # print(files[i])
            scores.append(load_file(files[i]))

            test_average.append([ta() for ta in scores[i]["test_average"][0:90]])


            loss.append([l() for l in scores[i]["loss"]])
            last_tick.append(len(loss))
            number_of_observations.append(len(test_average[i]))




            # epochs = [i * step for i in range(1, number_of_observations + 1)]
            epochs.append([n for n in range(1, number_of_observations[i] + 1)])

            # fig, (per_class_ax) = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

            # average accuracy vs per-class accuracy
            # num_colors = len(test_per_class.keys()) + 1
            # cm = plt.get_cmap("Spectral")
            # per_class_ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])
    cy = cycler('color', ['black', 'red','green','yello','blue'])
    for i in range(5):
        if epochs[i] is not None:
            # plt.plot(epochs[i], test_average[i], marker='', linewidth=2)
            plt.plot(epochs[i], test_average[i], marker='', linewidth=2)


    plt.legend(['rPattern', 'cPattern','ImageNet', 'Random','non-watermark'], loc='lower right')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.ylabel("loss")
    # plt.title("Watermarked Federated MNIST Model (100 participants)")
    plt.show()

def plot_communication(scores: List[str]) -> None:
    # sns.set()
    scores = []
    test_average = []
    loss = []
    last_tick = []
    number_of_observations = []
    epochs = []
    step = 5
    for i in range(5):
        if files[i] is None:
            scores.append(0)
            test_average.append(0)
            number_of_observations.append(0)
            epochs.append(0)
        else:
            print(i)
            # print(files[i])
            scores.append(load_file(files[i]))

            test_average.append([ta() for ta in scores[i]["test_average"]])
            print(test_average)
            loss.append([l() for l in scores[i]["loss"]])
            last_tick.append(len(loss))
            number_of_observations.append(len(test_average[i]))

            # epochs = [i * step for i in range(1, number_of_observations + 1)]
            if i == 0 :
                print(scores[i]["epoch"])
                epochs.append([ta / 10 * (i+1) for ta in scores[i]["epoch"]])
                print(epochs)
            elif i == 1:
                epochs.append([ta/10*(i) for ta in scores[i]["epoch"]])
                print(epochs)
            elif i == 4:
                epochs.append([ta / 10 *(i)*5 for ta in scores[i]["epoch"]])
                print(epochs)
            else:
                epochs.append([ta / 10 *(i-1)*5 for ta in scores[i]["epoch"]])
                print(epochs)


    cy = cycler('color', ['grey','blue', 'red', 'green', 'yello'])
    mark1 = [128]
    mark2 = [40]
    mark3 = [35]
    mark4 = [28]
    mark5 = [121]
    print(epochs[0][0])
    print(test_average[0][0])
    plt.plot(epochs[0][15:400], test_average[0][15:400],  linestyle='--', marker = 'D', markevery = mark5, linewidth=2)
    plt.plot(epochs[1][15:700], test_average[1][15:700],  linestyle='--', marker = 'D', markevery = mark1, linewidth=2)
    plt.plot(epochs[2][4:140], test_average[2][4:140], linestyle='--',  marker = 'D', markevery = mark2, linewidth=2)
    plt.plot(epochs[3][3:70], test_average[3][3:70], linestyle='--', marker = 'D', markevery = mark3, linewidth=2)
    plt.plot(epochs[4][2:35], test_average[4][2:35], linestyle='--',  marker = 'D', markevery = mark4,linewidth=2)

    plt.legend(['Non-WM (1 $e_u$)','WMed (1 $e_u$)', 'WMed (5 $e_u$)', 'WMed (10 $e_u$)', 'WMed (20 $e_u$)'])
    plt.xlabel("$e_f$")
    plt.ylabel("Accuracy")
    plt.show()

def plot_watermark(scores: List[str]) -> None:
    # sns.set()
    # sns.set()
    scores = []
    test_average = []
    loss = []
    last_tick = []
    number_of_observations = []
    epochs = []
    step = 5
    for i in range(5):
        if files[i] is None:
            scores.append(0)
            test_average.append(0)
            number_of_observations.append(0)
            epochs.append(0)
        else:
            print(i)
            # print(files[i])
            scores.append(load_file(files[i]))


            for idx, ta in enumerate(scores[i]["retrain_rounds"]):

                if ta == 0:
                    scores[i]["retrain_rounds"].pop(idx)
                    scores[i]["retrain_rounds"].insert(idx, 100)

            test_average.append([ta for ta in scores[i]["retrain_rounds"][0:25]])

            # test_per_class = defaultdict(list)
            print(scores[i]["retrain_rounds"])

            loss.append([l() for l in scores[i]["loss"]])
            last_tick.append(len(loss))
            number_of_observations.append(len(test_average[i]))

            # epochs = [i * step for i in range(1, number_of_observations + 1)]
            epochs.append([n for n in range(1, number_of_observations[i] + 1)])

            # fig, (per_class_ax) = plt.subplots(nrows=1, ncols=1, constrained_layout=True)

            # average accuracy vs per-class accuracy
            # num_colors = len(test_per_class.keys()) + 1
            # cm = plt.get_cmap("Spectral")
            # per_class_ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])
    cy = cycler('color', ['black', 'red', 'green', 'yello', 'blue'])
    for i in range(5):
        if epochs[i] is not None:
            # plt.plot(epochs[i], test_average[i], marker='', linewidth=2)
            plt.plot(epochs[i], test_average[i], marker='', linewidth=2)

    plt.legend(['rPattern', 'mPattern','ImageNet', 'Random'], loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Retrain Rounds")
    # plt.ylabel("loss")
    # plt.title("Watermarked Federated MNIST Model (100 participants)")
    plt.show()


def plot_attack(files: List[str]) -> None:
    scores = []
    test_average = []
    sparsity_rate = []
    watermark_accuracy = []
    for i in range(4):
        # if files[i] is None:
        #     scores.append(0)
        #     test_average.append(0)
        #     number_of_observations.append(0)
        #     epochs.append(0)
        # else:

            print(files[i])
            scores.append(load_file(files[i]))
            #MNIST
            scores[i]["test_watermark"].insert(10, score.FloatScore(100.0))
            scores[i]["sparsity_rate"].insert(10, 0)
            watermark_accuracy.append([ta() for ta in scores[i]["test_watermark"][10:350]])
            test_average.append([ta() for ta in scores[i]["test_average"][9:350]])
            sparsity_rate.append([ta*100 for ta in scores[i]["sparsity_rate"][10:350]])
            #CIFAR
            # scores[i]["sparsity_rate"][0] = "100.0000"
            # watermark_accuracy.append([ta() for ta in scores[i]["test_watermark"]])
            # test_average.append([ta() for ta in scores[i]["test_average"]])
            # sparsity_rate.append([(100-float(ta[:-5])) for ta in scores[i]["sparsity_rate"]])

    color = ['orange', 'red', 'green', 'sienna']
    label = ['uPattern', 'rPattern', 'unRelated', 'unStructured']

    # ax2 = ax1.twinx()
    for i in range(4):
        plt.plot(sparsity_rate[i], test_average[i], color= color[i], marker='', linewidth=2, label = label[i])
        plt.plot(sparsity_rate[i], watermark_accuracy[i], color= color[i], linestyle='--', linewidth=2)
    plt.plot([x for x in range(100)], [53 for i in range(100)], color='grey',linestyle='dashdot', linewidth=2, label = "Threshold")

    plt.xlabel("Pruning rate")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    # plt.title("Watermarked Federated MNIST Model (100 participants)")
    plt.show()

def plot_middle(files: List[str]) -> None:
    scores = []
    test_average = []
    ma_test_average = []
    epochs = []
    watermark_accuracy = []
    ma_watermark_accuracy = []
    number_of_observations = []

    for i in range(4):
        if files[i] is None:
            scores.append(0)
            test_average.append(0)
            number_of_observations.append(0)
            # epochs.append(0)
        elif i == 0:

            print(files[i])
            scores.append(load_file(files[i]))

            ma_watermark_accuracy.append([ta() for ta in scores[i]["ma_test_watermark"]])
            watermark_accuracy.append([ta() for ta in scores[i]["test_watermark"]])
            ma_test_average.append([ta() for ta in scores[i]["ma_test_accuracy"]])
            test_average.append([ta() for ta in scores[i]["test_average"]])
            number_of_observations.append(len(test_average[i]))

            # epochs = [i * step for i in range(1, number_of_observations + 1)]
            epochs.append([n for n in range(1, number_of_observations[i] + 1)])

            print(len(watermark_accuracy[0]))
            print(len(test_average[0]))
            print(len(epochs[0]))
            # scores[i]["sparsity_rate"][0] = "100.0000"
            # watermark_accuracy.append([ta() for ta in scores[i]["test_watermark"]])
            # test_average.append([ta() for ta in scores[i]["test_average"]])
            # sparsity_rate.append([(100-float(ta[:-5])) for ta in scores[i]["sparsity_rate"]])
        elif i == 1:
            scores.append(load_file(files[i]))
            watermark_accuracy.append([ta() for ta in scores[i]["test_watermark"]])
            test_average.append([ta() for ta in scores[i]["test_average"]])
            number_of_observations.append(len(test_average[i]))

            # epochs = [i * step for i in range(1, number_of_observations + 1)]
            epochs.append([n for n in range(1, number_of_observations[i] + 1)])

        color = ['orange', 'red', 'sienna','green' ]
        label = ['local acc', 'local WM','global acc', 'global WM', 'origin global acc', 'origin global WM']

    # ax2 = ax1.twinx()

    plt.plot(epochs[0][0:60], ma_test_average[0][0:60], color= color[0], marker='', linewidth=2, label = label[0])
    plt.plot(epochs[0][0:60], ma_watermark_accuracy[0][0:60], color= color[0], linestyle='--', linewidth=2, label = label[1])
# plt.plot([x for x in range(100)], [47 for i in range(100)], color='grey',linestyle='dashdot', linewidth=2, label = "Threshold")
    plt.plot(epochs[0][0::2], test_average[0][0::2], color=color[1], marker='', linewidth=2, label = label[2])
    plt.plot(epochs[0][0::2], watermark_accuracy[0][0::2], color=color[1], linestyle='--', linewidth=2, label= label[3])

    plt.plot(epochs[1][0:60], test_average[1][0:60], color=color[2], marker='', linewidth=2, label = label[4])
    plt.plot(epochs[1][0:60], watermark_accuracy[1][0:60], color=color[2], linestyle='--', linewidth=2, label = label[5])

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    # plt.title("Watermarked Federated MNIST Model (100 participants)")
    plt.show()

def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_file1",
        type=str,
        default="newrecord/train_middlecifar_to_pattern_ws100_vgg16_20c_P_R1.pt2019-12-20.txt",
        help="Results data file for plotting."
    )
    parser.add_argument(
        "--results_file2",
        type=str,
        default="newrecord/trainingcifar_to_pattern_ws100_vgg16_20c_P_R1.pt2019-10-31.txt",
        help="Results data file for plotting."
    )
    parser.add_argument(
        "--results_file3",
        type=str,
        default=None,
        help="Results data file for plotting."
    )
    parser.add_argument(
        "--results_file4",
        type=str,
        default=None,
        help="Results data file for plotting."
    )
    parser.add_argument(
        "--results_file5",
        type=str,
        default= None,
        # default="newrecord/trainingfederated_mnist_l5_100c.pt2019-10-19.txt",
        help="Results data file for plotting."
    )
    parser.add_argument(
        "--watermark",
        default= False,
        action="store_true",
        help="Produce watermark plots."
    )
    parser.add_argument(
        "--training",
        default=False,
        action="store_true",
        help="Produce training plots."
    )
    parser.add_argument(
        "--attack",
        default=False,
        action="store_true",
        help="Produce attacker plots."
    )
    parser.add_argument(
        "--communication",
        default=False,
        action="store_true",
        help="Produce communication plots."
    )
    parser.add_argument(
        "--middle",
        default=True,
        action="store_true",
        help="Produce communication plots."
    )

    args = parser.parse_args()
    # if args.results_file is None:
    #     raise ValueError("Results data file must be provided.")

    return args


if __name__ == "__main__":
    args = handle_args()
    if args.training:
        plot = plot_training
    elif args.watermark:
        plot = plot_watermark
    elif args.attack:
        plot = plot_attack
    elif args.communication:
        plot = plot_communication
    elif args.middle:
        plot = plot_middle

    else:
        raise ValueError("You should not be here ¯\_(ツ)_/¯")
    files =[args.results_file1, args.results_file2, args.results_file3, args.results_file4, args.results_file5]
    plot(files)
