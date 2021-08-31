from collections import namedtuple, OrderedDict
from typing import List, Dict, Tuple
import Network
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import logger
import score
import copy
import updateWeight
import syft as sy
import sys
import pruning
import imagen as ig
import numpy as np
import numbergen as ng
import matplotlib.pyplot as plt
import matplotlib
import random
from PIL import Image
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision as tv

import time
from typing import List, Tuple, Any, Dict
import torch.utils.data as data
import math


log = logger.Logger(prefix=">>>")

from sys import getsizeof, stderr
from itertools import chain

from collections import deque
from decimal import Decimal


try:
    from reprlib import repr
except ImportError:
    pass
    
class SimpleDataset(data.Dataset):
    def __init__(self, data, labels) -> None:
        self.data, self.labels = data, labels
        self.count = len(self.labels)

    def __getitem__(self, index: int) -> (Any, int):
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return self.count

class UnNormalize(object):
    def __init__(self, mean, std, len):
        self.mean = mean
        self.std = std
        self.len = len

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if self.len == 1:
            tensor.mul_(self.std).add_(self.mean).mul_(255.0)
        else:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
            tensor.mul_(255.0)
        #tensor = torch.round(tensor).clamp(0, 255)
        return tensor

class Normalize(object):
    def __init__(self, mean, std, len):
        self.mean = mean
        self.std = std
        self.len = len

    def __call__(self, tensor):
        if self.len == 1:
            tensor.div_(255.0).sub_(self.mean).div_(self.std)
        else:
            tensor.div_(255.0)
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return tensor

def save_pattern(pattern, mask, y_target, n_adversaries, fl_model_name, repeat, img_filename_template):

    # create result dir
    REVERSE_TRIGGERS_DIR = "data/reverse_triggers_" + fl_model_name + '_' + str(repeat) + 'r_' + str(n_adversaries) + '_clients'
    save_dir = REVERSE_TRIGGERS_DIR
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if 'vgg16' in fl_model_name:
        save_color = 'RGB'
    else:
        save_color = 'L'

    img_filename = ('%s/%s' % (save_dir, img_filename_template % ('pattern', y_target)))
    img = Image.fromarray(pattern, save_color)
    img.save(img_filename, 'png')

    img_filename = ('%s/%s' % (save_dir, img_filename_template % ('mask', y_target)))
    img = Image.fromarray(mask * 255, save_color)
    img.save(img_filename, 'png')

    fusion = np.multiply(pattern, mask)
    #fusion = np.multiply(pattern, np.expand_dims(mask, axis=0))
    img_filename = ('%s/%s' % (save_dir, img_filename_template % ('fusion', y_target)))
    img = Image.fromarray(fusion, save_color)
    img.save(img_filename, 'png')

    pass
    
def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers) # user handlers take precedence
    seen = set()                  # track which object id's have already been seen
    default_size = getsizeof(0)   # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen: # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

class Experiment(object):
    def __init__(self, environment):
        self.training_ops = environment.training_ops
        self.test_ops = environment.test_ops
        self.watermark_ops = environment.watermark_ops
        self.model_architecture = self.training_ops.model_architecture
        self.federated_model = self.training_ops.federated_model
        self.fl_model_name = self.training_ops.fl_model_name
        self.checkpoint_save_path = self.training_ops.checkpoint_save_path
        self.client_num = self.training_ops.client_num
        self.use_cuda = self.training_ops.use_cuda
        self.resume_from_checkpoint_path = self.training_ops.resume_from_checkpoint_path
        self.weight_decay = self.watermark_ops.weight_decay
        self.watermark_loader = self.watermark_ops.watermark_loader
        self.pretrain = self.watermark_ops.pretrain
        self.w_retrain = self.watermark_ops.w_retrain
        self.epochs = self.training_ops.epochs
        self.train_set = self.training_ops.train_set
        self.real_client_num = 10
        self.model_name = self.watermark_ops.model_name
        self.test_set = self.test_ops.test_set
        self.test_loader = self.test_ops.test_loader
        self.number_of_classes = self.test_ops.number_of_classes
        self.watermark_model = self.watermark_ops.watermark_model
        self.pretrain_epochs = self.watermark_ops.pretrain_epochs
        self.learning_rate = self.watermark_ops.learning_rate

class ExperimentTraining(Experiment):
    def __init__(self, environment: namedtuple) -> None:
        super(ExperimentTraining, self).__init__(environment)


    def federated_train(self, log_interval: int = None) -> (nn.Module, Dict[str, List[score.Score]]):
        scores = {
            "test_average": [],
            "test_per_class": [],
            "test_watermark": [],
            "retrain_rounds": [],
            "sparsity_rate": [],
            "test_accuracy": [],
            "ma_test_accuracy": [],
            "ma_test_watermark": [],
            "loss": [],
            "epoch": []
        }

        begin2 = time.time()
        models = []
        optimizers = []
        best_acc = 0
        client = []
        client_tuple = []

        hook = sy.TorchHook(torch)
        # Create virtual workers
        for i in range(self.client_num):
            client.append(sy.VirtualWorker(hook, id=i))
        # Create secure worker, which is an aggregator
        secure_worker = sy.VirtualWorker(hook, id="secure_worker")

        if self.model_architecture == "vgg16":
            federated_model = self.federated_model(pretrained=True)
        else:
            federated_model = self.federated_model()

        if self.use_cuda:
            federated_model = federated_model.cuda()
            
        #state_dict = torch.load(checkpoint_federated_cifar_vgg16_100c_20localround.pt.pth)
        #print("Loading state from: {}".format('data/models/oldermodels/' + self.training_ops.fl_model_name))
        
        # MB is total communication in bytes during the training process
        Mb = 0
        # Total watermark training rounds is set to zero
        total_wm_retrain_rounds = 0
        start_epoch = 0
        
        # If pretrain flag is true, add pretrain_epochs to watermark train rounds
        if self.pretrain:
            total_wm_retrain_rounds += self.pretrain_epochs
        watermark_epoch = 0
        checkpoint_load = False
        
        # If the experiment stopped due to time limit, memory limit or other reasons,  ...
        # ... open the commented block below and read statistics from the latest checkpoint 
        if os.path.exists('checkpoint/checkpoint_' + self.model_name + '.pth'):
            checkpoint = torch.load('checkpoint/checkpoint_' + self.model_name + '.pth')
            checkpoint_load = True
            start_epoch = checkpoint['epoch']
            Mb = checkpoint['communication']
            begin2 = checkpoint['time']
            total_wm_retrain_rounds = checkpoint['wm_retrain_rounds']
                     
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace("module.", "")#k[7:]  # remove module.
                new_state_dict[name] = v
            # load params
            federated_model.load_state_dict(new_state_dict)
            test_average, test_per_class = self.test_model(federated_model)
            watermark_score = self.test_watermark(federated_model)
            with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                file.write('Global model, Epoch:{}, Test Accuracy: {:.2f}'.format(start_epoch, test_average.value))
                file.write('Global model, Epoch:{}, Watermark Accuuracy: {}'.format(start_epoch, watermark_score.value))
                file.write('Global model, Epoch:{}, wm retrain rounds: {:.2f}'.format(start_epoch, total_wm_retrain_rounds))
                file.write('Global model, Epoch:{}, Mb: {}'.format(start_epoch, Mb))
                file.write('Global model, Epoch:{}, Mb: {}'.format(start_epoch, begin2))
                file.write('Global model, Epoch:{}, Test Accuracy: {:.2f}'.format(start_epoch, test_average.value))
                file.write('Global model, Epoch:{}, Watermark Accuuracy: {}'.format(start_epoch, watermark_score.value))
                file.write('Global model, Epoch:{}, wm retrain rounds: {:.2f}'.format(start_epoch, total_wm_retrain_rounds))
                file.write('Global model, Epoch:{}, Mb: {}'.format(start_epoch, Mb))
                file.write('Global model, Epoch:{}, Mb: {}'.format(start_epoch, begin2))
                
        # create optimizer and scheduler for watermark retrain
        watermark_optimizer = optim.SGD(federated_model.parameters(), lr=self.learning_rate/self.training_ops.local_epoch)
        #if self.model_architecture == "MNIST_L5":
        #    watermark_optimizer = optim.SGD(federated_model.parameters(), lr=self.learning_rate/20.0)
        #else:
        #    watermark_optimizer = optim.SGD(federated_model.parameters(), lr=self.learning_rate)
        #watermark_scheduler = torch.optim.lr_scheduler.StepLR(watermark_optimizer, step_size=50, gamma=0.1)
        
        # If there is a need to train global model with pretrain flag, train it.
        if self.pretrain and checkpoint_load == False:
            federated_model, scores = self.train_watermark(federated_model, log_interval = 100)
            watermark_score = self.test_watermark(federated_model)
            with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                file.write('Global model, pretrained watermark accuracy: {:.2f}\n'.format(watermark_score.value))

        # Divide the dataset randomly into workers   
        federated_train_loader = []
        splited_inputs, splited_labels = self.split_dataset(self.train_set, self.client_num)
        for input, label, i in zip(splited_inputs, splited_labels, range(self.client_num)):
         
            training_dataset = sy.BaseDataset(input, label).send(client[i])           
            train_loader = torch.utils.data.DataLoader(training_dataset, shuffle=True,
                                                           batch_size=50)
            federated_train_loader.append(train_loader)

        #for epoch in range(int(self.client_num*self.epochs/self.real_client_num)):
        for epoch in range(start_epoch, self.epochs):
            # updated_wi is set to zero in before starting to every epoch
            # updated_wi is filled with averaging over worker's weights 
            wi = 0
            for (name, param) in federated_model.named_parameters():
                wi += 1
                updated_w = [0 for i in range(wi)]
            
            train_losses = [0 for i in range(self.client_num)]
            train_accuracies = [0 for i in range(self.client_num)]
                     
            # Select a random subset for workers (10 out of 100)
            rand_client_subset = np.random.choice(self.client_num, self.real_client_num, replace=False)
            for client_i in (rand_client_subset):
                # Send federated model weights to workers
                client_model = federated_model.copy().send(client[client_i])
                # Create optimizer for wrokers
                if self.pretrain_epochs == 150 or self.model_architecture == 'vgg16': #use smaller lr in random noise
                    client_optimizer = torch.optim.SGD(params=client_model.parameters(), lr=0.01)
                else:
                    client_optimizer = torch.optim.SGD(params=client_model.parameters(), lr=0.1) #0.1
                
                client_model.train()
                correct = 0
                train_loss = 0
                for local_epoch in range(self.training_ops.local_epoch):
                    for data, target in federated_train_loader[client_i]:
                        if self.use_cuda:
                            data, target = data.cuda(), target.cuda()
                        client_optimizer.zero_grad()
                        #if epoch == 100:
                        #    for g in client_optimizer.param_groups:
                        #        g["lr"] = 0.001
                        output = client_model(data)

                        if self.model_architecture == "MNIST_L5":
                            loss = nn.functional.nll_loss(output, target)
                        else:
                            loss = nn.CrossEntropyLoss()(output, target)
                        loss.backward()
                        client_optimizer.step()

                        loss = loss.get() # get the client's loss back
                        train_loss += loss.item()
                        if local_epoch == self.training_ops.local_epoch-1:
                            with torch.set_grad_enabled(False):
                                output = client_model(data)
                                _, pred = torch.max(output.data, 1)
                                pred = pred.get()
                                target = target.get()
                                correct += pred.eq(target.view_as(pred)).sum().item()
                        
                    train_accuracies[client_i] = 100 * correct / len(federated_train_loader[client_i].dataset) 
                    train_losses[client_i] += train_loss / (len(federated_train_loader[client_i].dataset) * self.training_ops.local_epoch)

                # Print GPU usage to check if there is a memory leak
                print('GPU usage {}'.format(torch.cuda.memory_allocated()))  
                # Move function send client model parameters to aggregator (secure worker)
                client_model.move(secure_worker)         
                updated_w = updateWeight.calculate_weight(updated_w, self.real_client_num, self.client_num, client_model)
                # Increase communication in bytes with the size of updated_w ...
                # ... and multiply it with 2, since download+upload is the total communication 
                Mb += total_size(updated_w)#sys.getsizeof(updated_w)
                print('Worker:{} Train Epoch: {}, worker data size {}, worker avg loss: {:.3f}, worker train acc:{:.2f}'.format(client_i, 
                                                           epoch+1, 
                                                           len(federated_train_loader[client_i].dataset), 
                                                           train_losses[client_i], train_accuracies[client_i]))
                                                           
                with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                    file.write('Worker:{} Train Epoch: {}, worker data size {}, worker avg loss: {:.3f}, worker train acc:{:.2f}\n'.format(client_i, 
                                                           epoch+1, 
                                                           len(federated_train_loader[client_i].dataset), 
                                                           train_losses[client_i], train_accuracies[client_i]))

                del client_model
                torch.cuda.empty_cache()

            # Receive data in bytes from secure worker to central server
            Mb += total_size(updated_w)
            # Update weights of federated model with average weights
            federated_model = updateWeight.send_weight(federated_model, updated_w)
            # Average loss of workers 
            avg_loss = np.sum(train_losses)/self.client_num
            scores["loss"].append(score.FloatScore(avg_loss))
            # Calculate watermark score after each epoch
            watermark_score = self.test_watermark(federated_model) 
            test_average, test_per_class = self.test_model(federated_model)  
            with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                file.write('Global model, Epoch:{}, Test Acc before watermark retrain: {}\n'.format(epoch+1, test_average.value))
                file.write('Global model, Epoch:{}, Watermark Acc before watermark retrain: {}\n'.format(epoch+1, watermark_score.value))

            # If retrain flag is set true, retrain federated model with watermark in each epoch
            if self.w_retrain:
                for e in range(100):
                    if watermark_score.value >= 98:
                        with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                            file.write('Global model, watermark threshold is reached, watermark accuracy: {:.2f}\n'.format(watermark_score.value))
                        break
                    federated_model.train()
                    for batch_idx, (data, target) in enumerate(self.watermark_loader, 0):
                        if self.use_cuda:
                            data, target = data.cuda(), target.cuda()
                        watermark_optimizer.zero_grad()
                        #if total_wm_retrain_rounds == 100:
                        #    for g in watermark_optimizer.param_groups:
                        #        g["lr"] = 0.001
                        output = federated_model(data)
                        if self.model_architecture == "MNIST_L5":
                            loss = nn.functional.nll_loss(output, target)
                        else:
                            loss = nn.CrossEntropyLoss()(output, target)
                        loss.backward()
                        watermark_optimizer.step()
                    print("retrain watermark {} rounds".format(e + 1))
                    watermark_score = self.test_watermark(federated_model)
                    total_wm_retrain_rounds += 1
                    with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                        file.write('Global model, watermark trained epochs: {}, watermark accuracy: {:.2f}\n'.format((e+1), watermark_score.value))

                # update scheduler after every retrain not in every epoch
                #watermark_scheduler.step()
            print("Testing at {}th epoch, train loss:{}".format(epoch+1, avg_loss))
            test_average, test_per_class = self.test_model(federated_model)
            with open('./train_record/new_epoch_logs_'+self.model_name+'.txt', 'a+') as file:
                file.write('Global model, Epoch:{}, Average_loss: {:.3f}, Test Acc: {:.2f}\n'.format(epoch+1, avg_loss, test_average.value))
            if self.w_retrain:
                watermark_score = self.test_watermark(federated_model)
                scores["test_watermark"].append(watermark_score)
            #if test_average.value > best_acc:
            if epoch % 50 == 0:
                self.save_checkpoint({
                    'epoch': epoch+1,
                    'state_dict': federated_model.state_dict(),
                    'best_acc': best_acc,
                    'communication': Mb,
                    'time': (time.time() - begin2),
                    'wm_retrain_rounds': total_wm_retrain_rounds
                }, epoch)
                best_acc = test_average.value
                best_epo = epoch+1
            #print("Best accuracy:{} at {}th epoch\n".format(best_acc, best_epo))
            total_epoch = (epoch)*self.training_ops.local_epoch
            scores["epoch"].append(total_epoch)
            scores["test_average"].append(test_average)
            scores["retrain_rounds"].append(watermark_epoch)
            scores["test_per_class"].append(test_per_class)


        with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
            file.write('The total communication is: {}\n'.format(Mb))
        log.info('Finished training. The total communication is: {}, the total wm retrain rounds are: {}\n'.format(Mb, total_wm_retrain_rounds))
        return federated_model, scores


    def test_model(self, model) -> (score.FloatScore, score.DictScore):
        """Test the model on the test dataset."""
        # model.eval is used for ImageNet models, batchnorm or dropout layers will work in eval mode.
        model.eval()

        def test_average() -> score.FloatScore:
            correct = 0
            # with torch.no_grad():
            with torch.set_grad_enabled(False):
                for batch_idx, (data, target) in enumerate(self.test_loader, 0):

                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()
                    # print(data.shape)
                    output = model(data)

                    _, pred = torch.max(output.data, 1)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            total = len(self.test_loader.dataset)
            accuracy = 100 * correct / total
            log.info("Test accuracy: ({}/{}) {:.2f}%".format(correct, total, accuracy))

            return score.FloatScore(accuracy)

        def test_per_class() -> score.DictScore:
            class_correct = list(0. for _ in range(self.number_of_classes))
            class_total = list(0. for _ in range(self.number_of_classes))

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.test_loader, 0):

                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()

                    output = model(data)
                    _, pred = torch.max(output.data, 1)
                    c = (pred == target).squeeze()
                    for i in range(target.shape[0]):
                        label = target[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            log.info("Accuracy of the network on the {} test images (per-class):".format(len(self.test_loader.dataset)))

            per_class_accuracy = {}
            for i in range(self.number_of_classes):
                accuracy = 100 * class_correct[i] / (class_total[i] + 0.0001)
                per_class_accuracy[i] = accuracy
                print('Accuracy of %5s : %2d %%' % (
                    i, accuracy))

            return score.DictScore(per_class_accuracy)

        return test_average(), test_per_class()


    def save_checkpoint(self, state, epoch: int):
        dir_path = self.checkpoint_save_path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(state, os.path.join(dir_path, 'checkpoint_%s.pth' % (self.model_name)))


    def train_watermark(self, watermark_model, log_interval: int = None) -> (nn.Module, Dict[str, List[score.Score]]):


        #if self.use_cuda:
        #    watermark_model = self.watermark_model().cuda()
        #else:
        #    watermark_model = self.watermark_model()

        scores = {
            "test_average": [],
            "test_per_class": [],
            "test_watermark": [],
            "retrain_rounds": [],
            "sparsity_rate": [],
            "test_accuracy": [],
            "ma_test_accuracy": [],
            "ma_test_watermark": [],
            "loss": [],
            "epoch": []
        }
        optimizer = optim.SGD(watermark_model.parameters(), lr=self.learning_rate, momentum=0.5, weight_decay=self.weight_decay)
        watermark_model.train()
        for epoch in range(self.pretrain_epochs):
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.watermark_loader, 0):

                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output = watermark_model(data)
                if self.model_architecture == "MNIST_L5":
                    loss = nn.functional.nll_loss(output, target)
                else:
                    loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                log.info('[%d/%d, %5d] loss: %.3f' %
                            (epoch + 1, self.pretrain_epochs, batch_idx + 1, running_loss))
                #running_loss = 0.0


            log.info("Testing at {}".format(epoch + 1))
            # avg_score, per_class_score = self.test_model(watermark_model)
            watermark_score = self.test_watermark(watermark_model)
            with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                file.write('Epochs: {}, loss: {:.2f}, Accuracy: {}\n'.format(epoch, running_loss/100.0, watermark_score.value))
            # scores["test_average"].append(avg_score)
            # scores["test_per_class"].append(per_class_score)
            scores["test_watermark"].append(watermark_score)

        log.info('Finished training watermark.')

        return watermark_model, scores

    def after_train_watermark(self, log_interval: int = None):

        scores = {
            "test_average": [],
            "test_per_class": [],
            "test_watermark": [],
            "retrain_rounds": [],
            "sparsity_rate": [],
            "test_accuracy": [],
            "ma_test_accuracy": [],
            "ma_test_watermark": [],
            "loss": [],
            "epoch": []
        }

        #federated_model, scores = self.federated_train(log_interval=1000)
        #Network.save_state(federated_model, "./data/models/" + self.fl_model_name)
        if self.model_architecture == "vgg16":
            federated_model = self.federated_model(pretrained=True)
        else:
            federated_model = self.federated_model()
            
        if self.use_cuda:
            watermark_model = self.federated_model().cuda()
            
        Network.load_state(watermark_model, "./data/models/" + self.fl_model_name)
        self.test_model(watermark_model)
        if self.model_architecture == "MNIST_L5":
            watermark_optimizer = optim.SGD(watermark_model.parameters(), lr=self.learning_rate/20.0)
        else:
            watermark_optimizer = optim.SGD(watermark_model.parameters(), lr=self.learning_rate)

        for epoch in range(100):
            running_loss = 0.0
            # criterion = nn.CrossEntropyLoss().cuda()
            for batch_idx, (data, target) in enumerate(self.watermark_ops.watermark_loader, 0):

                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                watermark_optimizer.zero_grad()
                output = watermark_model(data)
                if self.model_architecture == "MNIST_L5":
                    loss = nn.functional.nll_loss(output, target)
                else:
                    loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                watermark_optimizer.step()

                running_loss += loss.item()

                log.info('[%d/%d, %5d] loss: %.3f' %
                            (epoch + 1, 200, batch_idx + 1, running_loss))
                running_loss = 0.0

            # scores["loss"].append(score.FloatScore(running_loss))


            log.info("Testing at {}".format(epoch + 1))
            avg_score, per_class_score = self.test_model(watermark_model)
            watermark_score = self.test_watermark(watermark_model)
            with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                file.write('Epochs: {}, Accuracy: {}\n'.format(epoch, watermark_score.value))
            scores["test_average"].append(avg_score)
            scores["test_per_class"].append(per_class_score)
            scores["test_watermark"].append(watermark_score)

        log.info('Finished training watermark.')

        return watermark_model, scores

    def test_watermark(self, model: nn.Module) -> score.FloatScore:
        # model.eval is used for ImageNet models, batchnorm or dropout layers will work in eval mode.
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(self.watermark_loader, 0):
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                output = model(data)
                _, pred = torch.max(output.data, 1)

                total += target.size(0)
                correct += (pred == target).sum().item()

        accuracy = 100 * correct / total
        log.info("Accuracy of the network on the {} watermark images ({}/{}): {}".format(total, correct, total, accuracy))
        return score.FloatScore(accuracy)

    def split_dataset(self, train_set: torch.utils.data.Dataset, client_num: int):

        # split the data to the number of clients
        torch.manual_seed(0)
        split_data = torch.utils.data.dataset.random_split(train_set, [int(len(train_set) / client_num) for i in range(client_num)])
        #data = random.sample(split_data, self.real_client_num)
        data = random.sample(split_data, self.client_num)
        # seperate the inputs and labels
        input_subsets = []
        label_subsets = []
        #for i in range(self.real_client_num):
        for i in range(self.client_num):
            train_inputs = []
            train_labels = []
            data_loader = torch.utils.data.DataLoader(data[i], shuffle=True)
            for i, (inputs, labels) in enumerate(data_loader, 0):
                inputs = inputs.squeeze(0)
                labels = labels.squeeze(0)

                train_inputs.append(inputs)
                train_labels.append(labels)

            input_tensor = torch.tensor([t.cpu().numpy() for t in train_inputs])
            label_tensor = torch.tensor(np.array([t.cpu().numpy() for t in train_labels]))

            input_subsets.append(input_tensor)
            label_subsets.append(label_tensor)

        return input_subsets, label_subsets

    def non_iid_split_dataset(self, train_set: torch.utils.data.Dataset, client_num: int):

        # split the data to the number of clients
        torch.manual_seed(0)

        num_shards, num_imgs = 200, 250
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(self.client_num)}
        idxs = np.arange(num_shards*num_imgs)
        # labels = dataset.train_labels.numpy()
        labels = []
        for i in range(len(train_set)):
            labels.append(train_set[i][1])
        #labels = np.array(train_set.train_labels)
        labels = np.array(labels)

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign
        for i in range(self.client_num):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        input_subsets = []
        label_subsets = []
        for i in range(self.client_num):
            train_inputs = []
            train_labels = []
            for j in range(len(dict_users[i])):
                train_inputs.append(train_set[int(dict_users[i][j])][0])
                train_labels.append(train_set[int(dict_users[i][j])][1])
            
            input_tensor = torch.tensor([t.cpu().numpy() for t in train_inputs])
            label_tensor = torch.tensor(np.array([t for t in train_labels]))

            input_subsets.append(input_tensor)
            label_subsets.append(label_tensor)
        return input_subsets, label_subsets


class ExperimentAttack(Experiment):
    def __init__(self, environment: namedtuple) -> None:
        super(ExperimentAttack, self).__init__(environment)

    def fine_tuning_attack(self, n_adversaries:int, log_interval: int = None) -> (nn.Module):

        #client = []
        #hook = sy.TorchHook(torch)
        ## Create virtual workers
        #for i in range(self.client_num):
        #    client.append(sy.VirtualWorker(hook, id=i))
        ## Create secure worker, which is an aggregator
        #secure_worker = sy.VirtualWorker(hook, id="secure_worker")

        if self.model_architecture == "vgg16":
            client_model = self.federated_model(pretrained=True)
        else:
            client_model = self.federated_model()

        Network.load_state(client_model, "./data/watermark_model/"+self.model_name)        
        self.test_watermark(client_model)
        self.test_model(client_model, self.test_loader)
        
        # Divide the dataset randomly (but with the same seed) into workers   
        fine_tuning_data_loader = []
        splited_inputs, splited_labels = self.split_dataset(self.train_set, self.client_num)
        for input, label, i in zip(splited_inputs, splited_labels, range(self.client_num)):     
            training_dataset = SimpleDataset(input, label)       
            data_loader = torch.utils.data.DataLoader(training_dataset, shuffle=True,batch_size=50)
            fine_tuning_data_loader.append(data_loader)
        
        if self.pretrain_epochs == 150 or self.model_architecture == 'vgg16': #use smaller lr in random noise
            client_optimizer = torch.optim.SGD(params=client_model.parameters(), lr=0.01)
        else:
            client_optimizer = torch.optim.SGD(params=client_model.parameters(), lr=0.1)
            
        train_loss = 0
        # say that first client is the adversary
        for local_epoch in range(self.training_ops.local_epoch):
            client_model.train()
            for client_i in range(n_adversaries):
                for data, target in fine_tuning_data_loader[client_i]:
                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()
                    client_optimizer.zero_grad()
                    output = client_model(data)

                    if self.model_architecture == "MNIST_L5":
                        loss = nn.functional.nll_loss(output, target)
                    else:
                        loss = nn.CrossEntropyLoss()(output, target)
                    loss.backward()
                    client_optimizer.step()
                    train_loss += loss.item()/(len(fine_tuning_data_loader[client_i].dataset) * self.training_ops.local_epoch)
                    
            # Calculate test acc and watermark score after each epoch
            test_average, test_per_class = self.test_model(client_model,self.test_loader)
            watermark_score = self.test_watermark(client_model) 
            with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                file.write('Finetuning, num of adv:{}, Epoch:{}, Average loss: {:.2f}, Test Acc: {}\n'.format(n_adversaries, local_epoch+1,train_loss, test_average.value))
                file.write('Finetuning, num of adv:{}, Epoch:{}, Watermark Acc: {}\n'.format(n_adversaries, local_epoch+1, watermark_score.value))

        log.info('Finished fine tuning attack.')
        return client_model

    def pruning_attack(self, n_adversaries: int, log_interval: int = None) -> (nn.Module):
            
        if self.model_architecture == "vgg16":
            client_model = self.federated_model(pretrained=True)
        else:
            client_model = self.federated_model()

        Network.load_state(client_model, "./data/watermark_model/"+self.model_name)        
        self.test_watermark(client_model)
        self.test_model(client_model, self.test_loader)
        
        # Divide the dataset randomly (but with the same seed) into workers   
        pruning_data_loader = []
        splited_inputs, splited_labels = self.split_dataset(self.train_set, self.client_num)
        for input, label, i in zip(splited_inputs, splited_labels, range(self.client_num)):     
            training_dataset = SimpleDataset(input, label)       
            data_loader = torch.utils.data.DataLoader(training_dataset, shuffle=True,batch_size=50)
            pruning_data_loader.append(data_loader)
        
        
        if self.pretrain_epochs == 150 or self.model_architecture == 'vgg16': #use smaller lr in random noise
            client_optimizer = torch.optim.SGD(params=client_model.parameters(), lr=0.01)
        else:
            client_optimizer = torch.optim.SGD(params=client_model.parameters(), lr=0.1)

        
        pruning_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for level in pruning_levels:
            model_local = copy.deepcopy(client_model)
            model_local.eval()
            # parameters_to_prune = model_local.parameters()        
            for _, module in model_local.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=level)
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=level)

            train_loss = 0
            model_local.train()
            for local_epoch in range(self.training_ops.local_epoch):
                for client_i in range(n_adversaries):
                    for data, target in pruning_data_loader[client_i]:
                        if self.use_cuda:
                            data, target = data.cuda(), target.cuda()
                        client_optimizer.zero_grad()
                        output = model_local(data)
                        if self.model_architecture == "MNIST_L5":
                            loss = nn.functional.nll_loss(output, target)
                        else:
                            loss = nn.CrossEntropyLoss()(output, target)
                        loss.backward()
                        client_optimizer.step()
                        train_loss += loss.item()/(len(pruning_data_loader[client_i].dataset) * self.training_ops.local_epoch)
                    
                # Calculate test acc and watermark score after each epoch
                test_average, test_per_class = self.test_model(model_local,self.test_loader)
                watermark_score = self.test_watermark(model_local) 
            with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                file.write('Finepruning, pruning_level:{:.2f}, num of adv:{}, Epoch:{}, Average loss: {:.2f}, Test Acc: {}\n'.format(level, n_adversaries, local_epoch+1,train_loss, test_average.value))
                file.write('Finepruning, pruning_level:{:.2f}, num of adv:{}, Epoch:{}, Watermark Acc: {}\n'.format(level, n_adversaries, local_epoch+1, watermark_score.value))

        log.info('Finished fine pruning attack.')
        return client_model
        
    def evasion_attack(self, n_adversaries: int, log_interval: int = None, ood_dataset: data.Dataset = None) -> (nn.Module):

        if self.model_architecture == "vgg16":
            detector_model = self.federated_model(pretrained=True)
            Network.load_state(detector_model, "./data/watermark_model/"+self.model_name)  
            for param in detector_model.parameters():
                param.requires_grad = False
            n_features = detector_model.classifier[6].in_features
            detector_model.classifier[6] = nn.Sequential(nn.Linear(n_features, 1), nn.Sigmoid()) 
            input_size = 32
        else:
            detector_model = Network.MNIST_Detector()
            input_size = 28

        if ood_dataset == None:
            data_dir = 'data/datasets/tiny-imagenet-200/'
            data_transform = tv.transforms.Compose([
                    tv.transforms.Grayscale(),
                    tv.transforms.Resize(input_size),
                    tv.transforms.CenterCrop(input_size),
                    tv.transforms.ToTensor(),
                    #tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
                    tv.transforms.Normalize([0.5], [0.5]) ])
            ood_dataset = tv.datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transform)

            # Divide the dataset randomly (but with the same seed) into workers   
            in_training_data_loader = []
            splited_inputs, splited_labels = self.non_iid_split_dataset(self.train_set, self.client_num)
            #splited_inputs, splited_labels = self.split_dataset(self.train_set, self.client_num)
            for input, label, _ in zip(splited_inputs, splited_labels, range(self.client_num)): 
                in_label  =  torch.FloatTensor(len(input), 1).fill_(1.0)     
                training_dataset = SimpleDataset(input, in_label)     
                data_loader = torch.utils.data.DataLoader(training_dataset, shuffle=True,batch_size=50)
                in_training_data_loader.append(data_loader)

            splited_inputs, splited_labels = self.split_dataset(self.train_set, self.client_num)
            lengths = [len(training_dataset)*n_adversaries, 100, (len(ood_dataset)-len(training_dataset)*n_adversaries-100)]
            ood_data, ood_val_data, _ = torch.utils.data.dataset.random_split(ood_dataset, lengths)
            out_training_data_loader = torch.utils.data.DataLoader(ood_data, shuffle=True,batch_size=50)
            dataloader_iterator = iter(out_training_data_loader)
            out_val_data_loader = torch.utils.data.DataLoader(ood_val_data, shuffle=True,batch_size=50)

            in_val_data_loader = []
            splited_inputs, splited_labels = self.split_dataset(self.test_set, self.client_num)
            for input, label, _ in zip(splited_inputs, splited_labels, range(self.client_num)): 
                in_label  =  torch.FloatTensor(len(input), 1).fill_(1.0)     
                val_dataset = SimpleDataset(input, in_label)     
                data_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True,batch_size=50)
                in_val_data_loader.append(data_loader)

        del splited_inputs, splited_labels

        detector_optimizer = torch.optim.SGD(params=detector_model.parameters(), lr=0.01)
        print("Length of in-disribution training dataset {}".format(len(in_training_data_loader[0].dataset)*n_adversaries))
        print("Length of in-disribution val dataset {}".format(len(in_val_data_loader[0].dataset)*n_adversaries))
        print("Length of out-distribution training dataset {}".format(len(out_training_data_loader.dataset)))
        print("Length of out-distribution validation dataset {}".format(len(out_val_data_loader.dataset)))
        detector_model.train()
        for epoch in range(20):
            #print("Training evasion network, ... epoch {}".format(epoch))
            for client_i in range(n_adversaries):
                for data1, labels1 in in_training_data_loader[client_i]:
                    if self.use_cuda:
                        data1 = data1.cuda()
                    if self.model_architecture != "vgg16":
                        data1 = data1.view(-1, 28*28)
                    detector_optimizer.zero_grad()
                    output1 = detector_model(data1)
                    loss1 = -torch.mean(torch.log(output1))
                    try:
                        data2, _ = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(out_training_data_loader)
                        data2, _ = next(dataloader_iterator) 
                    if self.model_architecture != "vgg16":
                        data2 = data2.view(-1, 28*28)
                    if self.use_cuda:
                        data2 = data2.cuda()
                    output2 = detector_model(data2)
                    loss2 = -torch.mean(torch.log(1.0 - output2))
                    loss = loss1 + loss2
                    loss.backward()
                    detector_optimizer.step()

        detector_model.eval()
        th_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        for th in th_list:
            tpr = 0
            tnr = 0
            tpr_real = 0
            fpr_real = 0
            total_pos = 0
            total_neg = 0
            total_pos_real = 0
            total_neg_real = 0
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(self.test_loader, 0):
                    if self.use_cuda:
                        data = data.cuda()
                    if self.model_architecture != "vgg16":
                            data = data.view(-1, 28*28)
                    output = detector_model(data)
                    fpr_real +=  (output < th).sum().item()
                    total_neg_real += len(output)
                for client_i in range(n_adversaries):
                    for batch_idx, (data, _) in enumerate(in_val_data_loader[client_i], 0):
                        if self.use_cuda:
                            data = data.cuda()
                        if self.model_architecture != "vgg16":
                            data = data.view(-1, 28*28)
                        output = detector_model(data)
                        tnr +=  (output > th).sum().item()
                        total_neg += len(output)
                for batch_idx, (data, _) in enumerate(out_val_data_loader, 0):
                    if self.use_cuda:
                        data = data.cuda()
                    if self.model_architecture != "vgg16":
                            data = data.view(-1, 28*28)
                    output = detector_model(data)
                    tpr +=  (output < th).sum().item()
                    total_pos += len(output)
            #print("For threshold {}, tpr: {}, fpr: {}".format(th, tpr, fpr))
            for batch_idx, (data, _) in enumerate(self.watermark_loader, 0):
                if self.use_cuda:
                    data = data.cuda()
                if self.model_architecture != "vgg16":
                    data = data.view(-1, 28*28)
                output = detector_model(data)
                tpr_real +=  (output < th).sum().item()
                total_pos_real += len(output)
            tnr = 100 * tnr  / total_neg
            tpr = 100 * tpr / total_pos
            fpr_real = 100 * fpr_real / total_neg_real
            tpr_real = 100 * tpr_real / total_pos_real
            #if tnr >= (95.0) and tnr <= (96.0):
            print("For threshold {}, tnr real for val set: {}, fpr for test set {}, tpr for tinyimagenet set {}, tpr for wm set: {}".format(th, tnr, fpr_real, tpr, tpr_real))
            print("tinyimagenet set {}, wm set {}, total test set {}, client test set {}".format(total_pos, total_pos_real, total_neg, total_neg_real))
            #    break
            #else:
            #    th += 0.01 
            #with open('./train_record/epoch_logs_' + self.model_name + '.txt', 'a+') as file:
            #    file.write('Evasion attack, threshold:{}, num of adv:{}, TPR: {}, FPR {}\n'.format(th, n_adversaries, tpr, fpr))
            with open('./train_record/epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                file.write("For threshold {}, tnr real for val set: {}, fpr for test set {}, tpr for tinyimagenet set {}, tpr for wm set: {}\n".format(th, tnr, fpr_real, tpr, tpr_real))
        return detector_model
                   

    def neural_cleanse(self, n_adversaries: int, log_interval: int = None, patching_type: str = 'unlearn', repeat: int = 0) -> (nn.Module):

        #############################################
        ## Paramaters for neural cleanse algorithm ##
        #############################################
        print(patching_type)
        # if resetting cost to 0 at the beginning
        # default is true for full optimization, set to false for early detection
        self.reset_cost_to_zero = True
        self.init_cost = 1e-4  # initial weight used for balancing two objectives
        # min/max of mask
        self.mask_min = 0
        self.mask_max = 1
        # min/max of raw pixel intensity
        self.color_min = 0
        self.color_max = 255
        # total optimization iterations
        self.steps = 1000
        # threshold of attack success rate for dynamically changing cost
        self.attack_succ_threshold = 0.99
        # epsilon (keras epsilon) used in tanh
        self.epsilon = 1e-07
        # verbose level, 0, 1 or 2
        self.verbose = 2
        # early stop flag
        self.early_stop = True
        # patience
        self.patience = 10
        # early stop threshold
        self.early_stop_threshold = 0.99
        # early stop patience
        self.early_stop_patience = 2 * self.patience
        # multiple of changing cost, down multiple is the square root of this
        self.cost_multiplier_up = 2
        self.cost_multiplier_down = self.cost_multiplier_up ** 1.5
        # learning rate
        self.lr = 0.001
        # masked but clean samples rate used in patching via unlearning
        self.perc = 0.2

        if self.model_architecture == "vgg16":
            client_model = self.federated_model(pretrained=True)
            img_size = 32
            num_channels = 3
            self.norm  = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), len=num_channels)
            self.unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), len=num_channels)
        else:
            client_model = self.federated_model()
            img_size = 28
            num_channels = 1
            self.norm  = Normalize(mean=(0.5), std=(0.5), len=num_channels)
            self.unorm = UnNormalize(mean=(0.5), std=(0.5), len=num_channels)

        Network.load_state(client_model, "./data/watermark_model/"+self.model_name)        
        #self.test_watermark(client_model)
        #self.test_model(client_model, self.test_loader)
        
        # Divide the dataset randomly (but with the same seed) into workers   
        test_generator = []
        splited_inputs, splited_labels = self.split_dataset(self.train_set, self.client_num)
        for input, label, i in zip(splited_inputs, splited_labels, range(self.client_num)):     
            training_dataset = SimpleDataset(input, label)       
            data_loader = torch.utils.data.DataLoader(training_dataset, shuffle=True,batch_size=32)
            test_generator.append(data_loader)

        log_mapping = {}
        # y_label list to analyze
        y_target_list = list(range(self.number_of_classes))
        pattern_list = []
        mask_list = []
        for y_target in y_target_list:
            print('processing label %d' % y_target)

            visualize_start_time = time.time()
            pattern, mask, logs = self.visualize_trigger_w_mask(client_model, test_generator, y_target, img_size, num_channels, 
                                                                n_adversaries, save_pattern_flag=True)
            # meta data about the generated mask
            print('pattern, shape: %s, min: %f, max: %f' %(str(pattern.shape), torch.min(pattern), torch.max(pattern)))
            print('mask, shape: %s, min: %f, max: %f' %(str(mask.shape), torch.min(mask), torch.max(mask)))
            
            visualize_end_time = time.time()
            print('visualization cost %f seconds' %(visualize_end_time - visualize_start_time))
            log_mapping[y_target] = logs

            pattern_numpy = pattern.cpu().numpy()
            mask_numpy = mask.cpu().numpy()
            # if you want to save patterns, then comment out the below line
            #save_pattern(pattern_numpy, mask_numpy, y_target, n_adversaries, self.model_name, repeat, self.model_architecture + '_visualize_%s_label_%d.png')
            pattern_list.append(pattern)
            mask_list.append(mask)

        if patching_type == 'unlearn':
            for client_i in range(n_adversaries):
                len_ = len(test_generator[client_i].dataset.data)
                randomlen_ = int(len_ * self.perc * 0.1) #int((len_ * self.perc // self.client_num)*self.client_num)
                # create random indexes for each client
                # make sure the total masked data is equally divided between classes
                randomlen_ = int((randomlen_ // self.number_of_classes)*self.number_of_classes)
                reverse_patch_idx = np.random.randint(0, len_ -1, randomlen_ -1) 
                tpc = len(reverse_patch_idx)/self.number_of_classes
                for idx,i in enumerate(reverse_patch_idx):
                    ci = int(math.floor(idx/tpc))
                    test_generator[client_i].dataset.data[i-1] = (1-mask_list[ci]) * self.unorm(test_generator[client_i].dataset.data[i-1]) + mask_list[ci] * pattern_list[ci]
                    self.norm(test_generator[client_i].dataset.data[i-1])
            if self.pretrain_epochs == 150 or self.model_architecture == 'vgg16': #use smaller lr in random noise
                client_optimizer = torch.optim.SGD(params=client_model.parameters(), lr=0.01)
            else:
                client_optimizer = torch.optim.SGD(params=client_model.parameters(), lr=0.1)
                
            client_model.train()
            train_loss = 0
            test_average, test_per_class = self.test_model(client_model,self.test_loader)
            watermark_score = self.test_watermark(client_model) 
            with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                file.write('FL model, num of adv:{}, Test Acc: {}\n'.format(n_adversaries, test_average.value))
                file.write('FL model, num of adv:{}, Watermark Acc: {}\n'.format(n_adversaries, watermark_score.value))
            # retrain the model
            for local_epoch in range(self.training_ops.local_epoch):
                for client_i in range(n_adversaries):
                    for data, target in test_generator[client_i]:
                        if self.use_cuda:
                            data, target = data.cuda(), target.cuda()
                        client_optimizer.zero_grad()
                        output = client_model(data)

                        if self.model_architecture == "MNIST_L5":
                            loss = nn.functional.nll_loss(output, target)
                        else:
                            loss = nn.CrossEntropyLoss()(output, target)
                        loss.backward()
                        client_optimizer.step()
                        train_loss += loss.item()/(len(test_generator[client_i].dataset) * self.training_ops.local_epoch)

                # Calculate test acc and watermark score after each epoch
                test_average, test_per_class = self.test_model(client_model,self.test_loader)
                watermark_score = self.test_watermark(client_model) 
                with open('./train_record/new_epoch_logs_' + self.model_name + '.txt', 'a+') as file:
                    file.write('Neural cleanse with unlearning, num of adv:{}, Epoch:{}, Average loss: {:.2f}, Test Acc: {}\n'.format(n_adversaries, local_epoch+1,train_loss, test_average.value))
                    file.write('Neural cleanse with unlearning, num of adv:{}, Epoch:{}, Watermark Acc: {}\n'.format(n_adversaries, local_epoch+1, watermark_score.value))

        elif patching_type == 'evasion':
            if self.model_architecture == "vgg16":
                detector_model = self.federated_model(pretrained=True)
            else:
                detector_model = self.federated_model()

            Network.load_state(detector_model, "./data/watermark_model/"+self.model_name)   

            n_features = detector_model.classifier[6].in_features
            detector_model.classifier[6] = nn.Linear(n_features, 1)
            
            # Divide the dataset randomly (but with the same seed) into workers   
            training_data_loder = []
            splited_inputs, splited_labels = self.split_dataset(self.train_set, self.client_num)
            for input, label, i in zip(splited_inputs, splited_labels, range(self.client_num)):     
                training_dataset = SimpleDataset(input, label)       
                data_loader = torch.utils.data.DataLoader(training_dataset, shuffle=True,batch_size=50)
                training_data_loder.append(data_loader)
            
            detector_optimizer = torch.optim.SGD(params=detector_model.parameters(), lr=0.01)

        log.info('Finished neural cleanse attack.')
        return client_model

    def visualize_trigger_w_mask(self, client_model: nn.Module, 
                                    gen: torch.utils.data.DataLoader, 
                                    y_target: int, img_size: int, num_channels: int, 
                                    n_adversaries: int, save_pattern_flag: bool = True):

        # initialize with random mask
        pattern_init = np.random.random((num_channels, img_size, img_size)) * 255.0
        mask_init = np.random.random((img_size, img_size))
        print('resetting state')

        # setting cost
        if self.reset_cost_to_zero:
            self.cost = 0
        else:
            self.cost = self.init_cost

        # setting mask and pattern
        mask = np.array(mask_init)
        pattern = np.array(pattern_init)
        mask = np.clip(mask, self.mask_min, self.mask_max)
        pattern = np.clip(pattern, self.color_min, self.color_max)
        mask = np.expand_dims(mask, axis=0)

        # convert to tanh space
        mask_tanh = np.arctanh((mask - 0.5) * (2 - self.epsilon))
        pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - self.epsilon))
        print('mask_tanh', np.min(mask_tanh), np.max(mask_tanh))
        print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))

        self.mask_tanh_tensor = torch.tensor(mask_tanh, dtype=torch.float)
        self.pattern_tanh_tensor = torch.tensor(pattern_tanh, dtype=torch.float)
        self.mask_tanh_tensor.requires_grad = True
        self.pattern_tanh_tensor.requires_grad = True

        # resetting optimizer states
        self.optimizer = torch.optim.Adam([self.mask_tanh_tensor, self.pattern_tanh_tensor], lr=self.lr, betas=(0.5, 0.9))

        # best optimization results
        mask_best = None
        pattern_best = None
        reg_best = float('inf')

        # logs and counters for adjusting balance cost
        logs = []
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # counter for early stop
        early_stop_counter = 0
        early_stop_reg_best = reg_best
        train_accuracy = 0

        client_model.eval()

        # loop start
        for step in range(self.steps):
            correct = 0
            # record loss for all mini-batches
            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            loss_acc_list = []
            for client_i in range(n_adversaries):
                for data, _ in gen[client_i]:
                    target = torch.ones(len(data), dtype=torch.long)*y_target
                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()
                    pattern_raw_tensor = ((torch.tanh(self.pattern_tanh_tensor) / (2 - self.epsilon) + 0.5) * 255.0)
                    mask_raw_tensor = (torch.tanh(self.mask_tanh_tensor) / (2 - self.epsilon) + 0.5)
                    X_adv_raw_tensor = (1-mask_raw_tensor) * self.unorm(data) + mask_raw_tensor * pattern_raw_tensor
                    self.norm(X_adv_raw_tensor)
                    output = client_model(X_adv_raw_tensor)
                    if self.model_architecture == "MNIST_L5":
                        loss_ce = nn.functional.nll_loss(output, target)
                    else:
                        loss_ce = nn.CrossEntropyLoss()(output, target)

                    loss_reg = (torch.sum(torch.abs(mask_raw_tensor)) / num_channels)
                    loss = loss_ce + self.cost * loss_reg

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    output = client_model(X_adv_raw_tensor)
                    _, pred = torch.max(output.data, 1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    loss_ce_list.append(loss_ce)
                    loss_reg_list.append(loss_reg)
                    loss_list.append(loss)


            avg_loss_acc = correct / (n_adversaries*len(gen[client_i].dataset))
            avg_loss_ce = (torch.mean(torch.stack(loss_ce_list))).item()
            avg_loss_reg = (torch.mean(torch.stack(loss_reg_list))).item()
            avg_loss = (torch.mean(torch.stack(loss_list))).item()

            # check to save best mask or not
            if avg_loss_acc >= self.attack_succ_threshold and avg_loss_reg < reg_best:
                mask_best = mask_raw_tensor.detach()
                mask_best = mask_best[0,:,:]
                # meta data about the generated mask
                pattern_best = pattern_raw_tensor.detach()
                pattern_best = pattern_best[0,:,:]
                reg_best = avg_loss_reg

             # verbose
            if self.verbose != 0:
                if self.verbose == 2 or step % (self.steps // 10) == 0:
                    print('step: %3d, cost: %.2E, attack: %.3f, loss: %f, ce: %f, reg: %f, reg_best: %f' %
                          (step, Decimal(self.cost), avg_loss_acc, avg_loss,
                           avg_loss_ce, avg_loss_reg, reg_best))

            # save log
            logs.append((step,
                         avg_loss_ce, avg_loss_reg, avg_loss, avg_loss_acc,
                         reg_best, self.cost))

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= self.early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and
                        cost_up_flag and
                        early_stop_counter >= self.early_stop_patience):
                    print('early stop')
                    break

            # check cost modification
            if self.cost == 0 and avg_loss_acc >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    self.cost = self.init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2E' % Decimal(self.cost))
            else:
                cost_set_counter = 0

            if avg_loss_acc >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.verbose == 2:
                    print('up cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost * self.cost_multiplier_up)))
                self.cost *= self.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                if self.verbose == 2:
                    print('down cost from %.2E to %.2E' %
                          (Decimal(self.cost),
                           Decimal(self.cost / self.cost_multiplier_down)))
                self.cost /= self.cost_multiplier_down
                cost_down_flag = True  

        # save the final version
        if mask_best is None:
            mask_best = mask_raw_tensor.detach()
            mask_best = mask_best[0,:,:]
            # meta data about the generated mask
            pattern_best = pattern_raw_tensor.detach()
            pattern_best = pattern_best[0,:,:]

        return pattern_best, mask_best, logs

    def split_dataset(self, train_set: torch.utils.data.Dataset, client_num: int):

        # split the data to the number of clients
        torch.manual_seed(0)
        split_data = torch.utils.data.dataset.random_split(train_set, [int(len(train_set) / client_num) for i in range(client_num)])
        #data = random.sample(split_data, self.real_client_num)
        data = random.sample(split_data, self.client_num)
        # seperate the inputs and labels
        input_subsets = []
        label_subsets = []
        for i in range(self.client_num):
            train_inputs = []
            train_labels = []
            data_loader = torch.utils.data.DataLoader(data[i], shuffle=True)
            for i, (inputs, labels) in enumerate(data_loader, 0):
                inputs = inputs.squeeze(0)
                labels = labels.squeeze(0)

                train_inputs.append(inputs)
                train_labels.append(labels)

            input_tensor = torch.tensor([t.cpu().numpy() for t in train_inputs])
            label_tensor = torch.tensor(np.array([t.cpu().numpy() for t in train_labels]))

            input_subsets.append(input_tensor)
            label_subsets.append(label_tensor)
        return input_subsets, label_subsets

    def non_iid_split_dataset(self, train_set: torch.utils.data.Dataset, client_num: int):

        # split the data to the number of clients
        torch.manual_seed(0)

        num_shards, num_imgs = 200, 300#250
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([]) for i in range(self.client_num)}
        idxs = np.arange(num_shards*num_imgs)
        # labels = dataset.train_labels.numpy()
        labels = []
        for i in range(len(train_set)):
            labels.append(train_set[i][1])
        #labels = np.array(train_set.train_labels)
        labels = np.array(labels)

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign
        for i in range(self.client_num):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        input_subsets = []
        label_subsets = []
        for i in range(self.client_num):
            train_inputs = []
            train_labels = []
            for j in range(len(dict_users[i])):
                train_inputs.append(train_set[int(dict_users[i][j])][0])
                train_labels.append(train_set[int(dict_users[i][j])][1])
            
            input_tensor = torch.tensor([t.cpu().numpy() for t in train_inputs])
            label_tensor = torch.tensor(np.array([t for t in train_labels]))

            input_subsets.append(input_tensor)
            label_subsets.append(label_tensor)
        return input_subsets, label_subsets

    def test_model(self, model: nn.Module, test_loader: torch.utils.data.DataLoader) -> (score.FloatScore, score.DictScore):
        """Test the model on the test dataset."""
        # model.eval is used for ImageNet models, batchnorm or dropout layers will work in eval mode.
        model.eval()
        number_of_classes = self.test_ops.number_of_classes

        # if self.use_cuda:
        #     model = model.cuda()


        def test_average() -> score.FloatScore:
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader, 0):
                    # print(data.shape)
                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()

                    output = model(data)
                    _, predicted = torch.max(output.data, 1)

                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            log.info("Accuracy of the network on the {} test images (average): {}".format(total, accuracy))
            return score.FloatScore(accuracy)

        def test_per_class() -> score.DictScore:
            class_correct = list(0. for _ in range(number_of_classes))
            class_total = list(0. for _ in range(number_of_classes))
            total = 0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader, 0):

                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()

                    total += target.size(0)

                    output = model(data)
                    _, predicted = torch.max(output, 1)
                    c = (predicted == target).squeeze()
                    for i in range(target.shape[0]):
                        label = target[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1

            log.info("Accuracy of the network on the {} test images (per-class):".format(total))

            per_class_accuracy = {}
            for i in range(number_of_classes):
                accuracy = 100 * class_correct[i] / class_total[i]
                per_class_accuracy[i] = accuracy
                print('Accuracy of %5s : %2d %%' % (
                    i, accuracy))

            return score.DictScore(per_class_accuracy)

        return test_average(), test_per_class()

    def test_watermark(self, model: nn.Module) -> score.FloatScore:
        # model.eval is used for ImageNet models, batchnorm or dropout layers will work in eval mode.
        model.eval()
        # if self.use_cuda:
        #     model = model.cuda()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.watermark_loader, 0):
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                output = model(data)
                _, predicted = torch.max(output.data, 1)

                total += predicted.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / max(total, 1)
        log.info("Accuracy of the network on the {} watermark images ({}/{}): {}".format(total, correct, total, accuracy))
        return score.FloatScore(accuracy)



class GenerateWatermark(Experiment):
    def __init__(self, environment: namedtuple) -> None:
        super(GenerateWatermark, self).__init__(environment)


    def generate_mpattern(self, x_input, y_input, num_class, num_picures):
        x_pattern = int(x_input * 2 / 3. - 1)
        y_pattern = int(y_input * 2 / 3. - 1)

        for cls in range(num_class):
            # define patterns
            patterns = []
            patterns.append(
                ig.Line(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001, orientation=np.pi * ng.UniformRandom(),
                        x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, scale=0.8))
            patterns.append(
                ig.Arc(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001, orientation=np.pi * ng.UniformRandom(),
                       x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, size=0.33))

            pat = np.zeros((x_pattern, y_pattern))
            for i in range(6):
                j = np.random.randint(len(patterns))
                pat += patterns[j]()
            res = pat > 0.5
            pat = res.astype(int)
            print(pat)

            x_offset = np.random.randint(x_input - x_pattern + 1)
            y_offset = np.random.randint(y_input - y_pattern + 1)
            print(x_offset, y_offset)

            for i in range(num_picures):
                base = np.random.rand(x_input, y_input)
                base[x_offset:x_offset + pat.shape[0], y_offset:y_offset + pat.shape[1]] += pat
                d = np.ones((x_input, x_input))
                img = np.minimum(base, d)
                if not os.path.exists("./data/datasets/MPATTERN/" + str(cls) + "/"):
                    os.makedirs("./data/datasets/MPATTERN/" + str(cls) + "/")
                plt.imsave("./data/datasets/MPATTERN/" + str(cls) + "/wm_" + str(i + 1) + ".png", img, cmap=matplotlib.cm.gray)

    def generate_cpattern(self, x_input, y_input, num_class, num_picures):
        x_pattern = int(x_input * 2 / 3. - 1)
        y_pattern = int(y_input * 2 / 3. - 1)

        for cls in range(num_class):
            # define patterns
            patterns = []
            patterns.append(
                ig.Line(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001, orientation=np.pi * ng.UniformRandom(),
                        x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, scale=0.8))
            patterns.append(
                ig.Arc(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001, orientation=np.pi * ng.UniformRandom(),
                       x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, size=0.33))

            pat = np.zeros((x_pattern, y_pattern))
            for i in range(8):
                j = np.random.randint(len(patterns))
                pat += patterns[j]()
            res = pat > 0.5
            pat = res.astype(int)
            print(pat)

            x_offset = np.random.randint(x_input - x_pattern + 1)
            y_offset = np.random.randint(y_input - y_pattern + 1)
            print(x_offset, y_offset)
            random_num = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            print(random_num)

            for i in range(num_picures):
                im = np.zeros((32, 32, 3), dtype='uint8')
                print(im)
                for c in range(3):
                    base = np.random.rand(x_input, y_input) * 255
                    print(base[x_offset:x_offset + pat.shape[0], y_offset:y_offset + pat.shape[1]])
                    # base[x_offset:x_offset+pat.shape[0], y_offset:y_offset+pat.shape[1]] -= pat
                    # print(base[x_offset:x_offset+pat.shape[0], y_offset:y_offset+pat.shape[1]])
                    # d =
                    d = np.zeros((x_input, y_input))
                    # print(base[15,:])

                    # print(img[15,:])
                    base[x_offset:x_offset + pat.shape[0], y_offset:y_offset + pat.shape[1]] -= (pat * 255)
                    img = np.maximum(base, d)
                    img[x_offset:x_offset + pat.shape[0], y_offset:y_offset + pat.shape[1]] += (pat * random_num[c])
                    print(img[15, :])
                    im[:, :, c] = img
                print(im[:, 15, :])
                # imgs.append(img)

                image = Image.fromarray(im.astype(dtype='uint8'), 'RGB')
                if not os.path.exists("./data/datasets/CPATTERN/" + str(cls) + "/"):
                    os.makedirs("./data/datasets/CPATTERN/" + str(cls) + "/")
                image.save("./data/datasets/CPATTERN/" + str(cls) + "/wm_" + str(i + 1) + ".png")

    def cifar_with_pattern(self, x_input, y_input, num_class, num_picures):

        x_pattern = int(x_input * 2 / 3. - 1)
        y_pattern = int(y_input * 2 / 3. - 1)

        train_data, _ = torch.utils.data.dataset.random_split(self.train_set, (int(num_picures*2), len(self.train_set) - int(num_picures*2)))
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

        # define patterns
        patterns = []
        patterns.append(
            ig.Line(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001,
                    orientation=np.pi * ng.UniformRandom(),
                    x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, scale=0.8))
        patterns.append(
            ig.Arc(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001,
                   orientation=np.pi * ng.UniformRandom(),
                   x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, size=0.33))

        pat = np.zeros((x_pattern, y_pattern))
        for i in range(8):
            j = np.random.randint(len(patterns))
            pat += patterns[j]()
        res = pat > 0.5
        pat = res.astype(int)
        x_offset = np.random.randint(x_input - x_pattern + 1)
        y_offset = np.random.randint(y_input - y_pattern + 1)
        random_num = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        count = [0 for x in range(10)]
        for i, (data, target) in enumerate(data_loader):
            #idx = i%10
            idx = int(((target-1)%10).cpu().numpy())
            if target != idx:
                count[idx] += 1
                data = np.array(data).squeeze(0)

                data = np.transpose(data*255, (1, 2, 0))
                data = ((data - data.min()) * (1 / (data.max() - data.min())) * 255)
                # print(data)
                # image = Image.fromarray(data.astype(dtype='uint8'), 'RGB')
                # image.save("./data/datasets/CIFAR&PATTERN/" + str(idx) + "/wm_" + str(i + 1) + ".png")
                im = np.zeros((32, 32, 3), dtype='uint8')
                for c in range(3):
                    d = np.zeros((x_input, y_input),dtype='uint8')
                    data[x_offset:x_offset + pat.shape[0], y_offset:y_offset + pat.shape[1], c] -= (pat*255)
                    img = np.maximum(data[:,:,c], d)
                    img[x_offset:x_offset + pat.shape[0], y_offset:y_offset + pat.shape[1]] += (pat * random_num[c])
                    # print(img[15, :])
                    im[:, :, c] = img
                # print(im[:, 15, :])
                # imgs.append(img)
                if count[idx] <= 10 :
                    image = Image.fromarray(im.astype(dtype='uint8'), 'RGB')
                    if not os.path.exists("./data/datasets/CIFAR&PATTERN/" + str(idx) + "/"):
                        os.makedirs("./data/datasets/CIFAR&PATTERN/" + str(idx) + "/")
                    image.save("./data/datasets/CIFAR&PATTERN/" + str(idx) + "/wm_" + str(i + 1) + ".jpg")

    def mnist_with_pattern(self, x_input, y_input, num_class, num_picures):
        x_pattern = int(x_input * 2 / 3. - 1)
        y_pattern = int(y_input * 2 / 3. - 1)

        train_data, _ = torch.utils.data.dataset.random_split(self.train_set,
                                                              (int(num_picures*2), len(self.train_set) - int(num_picures*2)))
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

        # define patterns
        patterns = []

        patterns.append(
            ig.Line(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001,
                    orientation=np.pi * ng.UniformRandom(),
                    x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, scale=0.8))
        patterns.append(
            ig.Arc(xdensity=x_pattern, ydensity=y_pattern, thickness=0.001,
                   orientation=np.pi * ng.UniformRandom(),
                   x=ng.UniformRandom() - 0.5, y=ng.UniformRandom() - 0.5, size=0.33))

        pat = np.zeros((x_pattern, y_pattern))
        for i in range(8):
            j = np.random.randint(len(patterns))
            pat += patterns[j]()
        res = pat > 0.5
        pat = res.astype(int)
        x_offset = np.random.randint(x_input - x_pattern + 1)
        y_offset = np.random.randint(y_input - y_pattern + 1)

        count = [0 for x in range(10)]
        for i, (data, target) in enumerate(data_loader):
            #idx = i % 10
            idx = int(((target-1)%10).cpu().numpy())
            if target != idx:
                count[idx] += 1
                data = np.array(data).squeeze(0)
                data = np.array(data).squeeze(0)
                idx = int(((target-1)%10).cpu().numpy())
                d = np.zeros((x_input, y_input))
                data[x_offset:x_offset + pat.shape[0],
                y_offset:y_offset + pat.shape[1]] += pat
                img = np.minimum(data, d)
                if count[idx] <= 10:
                    if not os.path.exists("./data/datasets/MNIST&PATTERN/" + str(idx) + "/"):
                        os.makedirs("./data/datasets/MNIST&PATTERN/" + str(idx) + "/")
                    plt.imsave("./data/datasets/MNIST&PATTERN/" + str(idx) + "/wm_" + str(i + 1) + ".png", img, cmap=matplotlib.cm.gray)
