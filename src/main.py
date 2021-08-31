import argparse
import configparser
import time
import pickle
from typing import List, Dict
import datetime
import config_helper
import datapre
import experiment
import logger
import Network
import score
import torch
import os
torch.set_default_tensor_type(torch.cuda.FloatTensor)

log = logger.Logger(prefix=">>>")

def main(config: configparser.ConfigParser, exp: str, gen_wm: str, patching_type: str):
    env = datapre.prepare_data(config)
    #experiment.GenerateWatermark(env).cifar_with_pattern(x_input=32, y_input=32, num_class=10, num_picures=100)
    if gen_wm == "None":
        begin = time.time()
        if not os.path.exists("./train_record/"):
            os.makedirs("./train_record/")
        with open('./train_record/new_epoch_logs_'+config["WATERMARK"]["model_name"]+'.txt', 'a+') as file:
            file.write('\nbegin time: {}, file name: {}\n'.format(begin, args.config_file))

        experiments = {

            "training": experiment_training,
            "finetuning": experiment_attack,
            "pruning": experiment_attack,
            "neuralcleanse": experiment_attack,
            "evasion": experiment_attack,
        }
        agent = {

            "training": experiment.ExperimentTraining(env),
            "finetuning": experiment.ExperimentAttack(env),
            "pruning": experiment.ExperimentAttack(env),
            "neuralcleanse": experiment.ExperimentAttack(env),
            "evasion": experiment.ExperimentAttack(env),
        }

        experiments[exp](
                env,
                agent[exp],
                config["DEFAULT"]["scores_save_path"] + str(exp) + config["WATERMARK"]["model_name"],
                exp)

        with open('./train_record/new_epoch_logs_' + config["WATERMARK"]["model_name"] + '.txt', 'a+') as file:
            file.write('\nfinish time: {}, file name: {}\n'.format(time.time() - begin, args.config_file))
        print(time.time() - begin)
    elif gen_wm == "mpattern":
        experiment.GenerateWatermark(env).generate_mpattern(x_input=28, y_input=28, num_class=10, num_picures=100)
    elif gen_wm == "cpattern":
        experiment.GenerateWatermark(env).generate_cpattern(x_input=32, y_input=32, num_class=10, num_picures=100)
    elif gen_wm == "mnistpattern":
        experiment.GenerateWatermark(env).mnist_with_pattern(x_input=28, y_input=28, num_class=10, num_picures=100)
    elif gen_wm == "cifarpattern":
        experiment.GenerateWatermark(env).cifar_with_pattern(x_input=32, y_input=32, num_class=10, num_picures=100)

def experiment_training(env, agent, path_body: str, exp:str) -> None:

    if env.federated_retrain == 1:
        federated_model, scores = agent.after_train_watermark(log_interval= 10)
    else:
        federated_model, scores = agent.federated_train(log_interval=1000)


    date = datetime.datetime.today().strftime('%Y-%m-%d')

    save_scores(
        scores,
        path_body + date)
    if not os.path.exists(env.watermark_model_save_path):
        os.makedirs(env.watermark_model_save_path)
    Network.save_state(federated_model, env.federated_model_path)

def experiment_attack(env, agent, path_body: str, exp:str, patching_type: str = 'unlearn') -> None:

    if exp == "finetuning":
        for n_adversaries in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            for i in range(4):
                attacker_model = agent.fine_tuning_attack(n_adversaries=n_adversaries, log_interval=10)
    elif exp == "pruning":
        for n_adversaries in [1]:
            for i in range(4):
                attacker_model = agent.pruning_attack(n_adversaries=n_adversaries, log_interval=10)
    elif exp == "neuralcleanse":
        for n_adversaries in [1, 2, 5, 10, 20, 30, 40]:
            for i in range(4):
                attacker_model = agent.neural_cleanse(n_adversaries=n_adversaries, log_interval=10, patching_type=patching_type, repeat=i)
    elif exp == "evasion":
        for n_adversaries in [1, 2, 5, 10, 20, 30, 40, 50]:
            attacker_model = agent.evasion_attack(n_adversaries=n_adversaries, log_interval=10)
    else:
        raise ValueError("Unknown experiment.")

    date = datetime.datetime.today().strftime('%Y-%m-%d')
    if not os.path.exists("./data/attacker_model/"):
        os.makedirs("./data/attacker_model/")
    #Network.save_state(attacker_model, "./data/attacker_model/" + str(exp) + "_" + config["WATERMARK"]["model_name"])


def save_scores(scores_dict: Dict[str, List[score.Score]], file_path: str) -> None:
    with open(file_path + '.txt', 'wb') as f:
        pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)

def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="./configurations/mnist-to-pattern-ws100-l5-100c-P-R1.ini",
        help="Configuration file for the experiment.")

    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        help="Experiment that should be performed: training|finetuning|pruning|neuralcleanse|evasion|baseline."
    )

    parser.add_argument(
        "--generate_watermark",
        type=str,
        default="None",
        help="Generated watermark can be: mpattern|cpattern|mnistpattern|cifarpattern"
    )
    parser.add_argument(
        "--patching_type",
        type = str,
        default="unlearn",
        help="patching type for neural cleanse"
    )
    args = parser.parse_args()
    if args.config_file is None:
        raise ValueError("Configuration file must be provided.")
    if args.experiment not in ["training","finetuning", "pruning", "neuralcleanse", "evasion"]:
        raise ValueError("Unknown experiment.")

    return args


if __name__ == "__main__":

    args = handle_args()
    config = config_helper.load_config(args.config_file)
    config_helper.print_config(config)
    main(config, args.experiment, args.generate_watermark, args.patching_type)
