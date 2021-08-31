import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.transforms import BlendedGenericTransform

file = 'train_record/epoch_logs_cifar10_to_pattern_ws100_vgg16_100c-P_R20.pt.txt'
test_acc = np.zeros((11,2))
wm_acc   = np.zeros((11,2))
with open(file ,'r') as file_obj:
    for line in file_obj:
        if 'FinetuningTester,' in line and 'Test Acc' in line:
            line_arr = line.replace(':', ',').strip().split(',')
            epoch = int(line_arr[4])
            n_adv = int(int(line_arr[2])/40)
            if epoch%10 == 0:
                idx = int(epoch/10)
                test_acc[idx, n_adv] += float(line_arr[-1])
        elif 'FinetuningTester,' in line and 'Watermark Acc' in line:
            line_arr = line.replace(':', ',').strip().split(',')
            epoch = int(line_arr[4])
            n_adv = int(int(line_arr[2])/40)
            if epoch%10 == 0:
                idx = int(epoch/10)
                wm_acc[idx, n_adv] += float(line_arr[-1])
        elif ('FL model,' in line) and ('Test Acc:' in line):
            line_arr = line.split()
            test_acc[0,:] = np.float(line_arr[-1])*4
        elif ('FL model,' in line) and ('Watermark Acc:' in line):
            line_arr = line.split()
            wm_acc[0,:] = np.float(line_arr[-1])*4

test_acc = test_acc/4.0
wm_acc = wm_acc/4.0
print("Test acc with 1 adv")
print(test_acc[:,0])
print("WM acc with 1 adv")
print(wm_acc[:,0])
#print("Test acc with 40 adv")
#print(test_acc[:,1])
#print("WM acc with 40 adv")
#print(wm_acc[:,1])

test_acc = np.zeros((101))
wm_acc   = np.zeros((101))
with open(file ,'r') as file_obj:
    for line in file_obj:
        if 'Neural cleanse with unlearning,' in line and 'Test Acc' in line and 'Epoch:1,' in line:
            line_arr = line.replace(':', ',').strip().split(',')
            epoch = int(line_arr[4])
            n_adv = int(int(line_arr[2]))
            print(n_adv)
            test_acc[n_adv-1] += float(line_arr[-1])
        elif 'Neural cleanse with unlearning,' in line and 'Watermark Acc' in line and 'Epoch:1,' in line:
            line_arr = line.replace(':', ',').strip().split(',')
            epoch = int(line_arr[4])
            n_adv = int(int(line_arr[2]))
            print(n_adv)
            wm_acc[n_adv-1] += float(line_arr[-1])
        elif ('FL model,' in line) and ('Test Acc:' in line):
            line_arr = line.split()
            test_acc[-1] = np.float(line_arr[-1])*4 
        elif ('FL model,' in line) and ('Watermark Acc:' in line):
            line_arr = line.split()
            wm_acc[-1] = np.float(line_arr[-1])*4 

test_acc = test_acc/4.0
wm_acc = wm_acc/4.0
print("Test acc")
print(test_acc)
print("WM acc ")
print(wm_acc)


"""
FL model, num of adv:5, Test Acc: 86.2
FL model, num of adv:5, Watermark Acc: 99.0
Neural cleanse with unlearning, num of adv:5, Epoch:1, Average loss: 0.01, Test Acc: 82.36
Neural cleanse with unlearning, num of adv:5, Epoch:1, Watermark Acc: 35.0
Neural cleanse with unlearning, num of adv:5, Epoch:2, Average loss: 0.01, Test Acc: 82.07
Neural cleanse with unlearning, num of adv:5, Epoch:2, Watermark Acc: 29.0
Neural cleanse with unlearning, num of adv:5, Epoch:3, Average loss: 0.01, Test Acc: 83.93
Neural cleanse with unlearning, num of adv:5, Epoch:3, Watermark Acc: 20.0
Neural cleanse with unlearning, num of adv:5, Epoch:4, Average loss: 0.01, Test Acc: 83.83
Neural cleanse with unlearning, num of adv:5, Epoch:4, Watermark Acc: 34.0
Neural cleanse with unlearning, num of adv:5, Epoch:5, Average loss: 0.01, Test Acc: 84.32
Neural cleanse with unlearning, num of adv:5, Epoch:5, Watermark Acc: 49.0
Neural cleanse with unlearning, num of adv:5, Epoch:6, Average loss: 0.01, Test Acc: 84.19
Neural cleanse with unlearning, num of adv:5, Epoch:6, Watermark Acc: 47.0
Neural cleanse with unlearning, num of adv:5, Epoch:7, Average loss: 0.01, Test Acc: 84.18
Neural cleanse with unlearning, num of adv:5, Epoch:7, Watermark Acc: 44.0
Neural cleanse with unlearning, num of adv:5, Epoch:8, Average loss: 0.01, Test Acc: 84.29
Neural cleanse with unlearning, num of adv:5, Epoch:8, Watermark Acc: 53.0
Neural cleanse with unlearning, num of adv:5, Epoch:9, Average loss: 0.01, Test Acc: 84.32
Neural cleanse with unlearning, num of adv:5, Epoch:9, Watermark Acc: 52.0
Neural cleanse with unlearning, num of adv:5, Epoch:10, Average loss: 0.01, Test Acc: 84.35
Neural cleanse with unlearning, num of adv:5, Epoch:10, Watermark Acc: 51.0
"""