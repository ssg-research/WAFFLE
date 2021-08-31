import numpy as np
import torch

def multiple_weight(updated_w, client_num, model):
    for i, (name, param) in enumerate(model.named_parameters(), 0):
            updated_w[i] += (1.5 / client_num) * param

    return updated_w

def pruning_weight(updated_w, client_num, model, federated_model):
    zero_update = []
    for i, (name, param) in enumerate(federated_model.named_parameters(), 0):
        b, c, h, w = param.shape
        zero_update.append(np.zero((b, c, h, w)))
    for i, (name, param) in enumerate(model.named_parameters(), 0):
        pass

def update_weight(new, prev):
    with torch.set_grad_enabled(False):
        for name, param in prev.named_parameters():
            #     new.block[0].weight.data = prev.block[0].weight.data
            x = name.split(".")
            if len(x) == 3 :
                index = int(x[1])               
                result = getattr(new, str(x[0]))[index]
                getattr(result, x[2]).data = param
            else:
                result = getattr(new, str(x[0]))
                getattr(result, str(x[1])).data = param

    return new

def zero_weight(model):

    updated_w = []
    with torch.set_grad_enabled(False):
        for i, (name, param) in enumerate(model.named_parameters(), 0):
            updated_w.append(torch.zeros(param.shape))

        return updated_w

def calculate_weight(updated_w, client_subset, num_clients, model):
    with torch.set_grad_enabled(False):
        for i, (name, param) in enumerate(model.named_parameters(), 0):
            updated_w[i] += (client_subset * param.get())/num_clients

        return updated_w


def send_weight(model, updated_w):
    for i, (name, param) in enumerate(model.named_parameters(), 0):
        x = name.split(".")
        if len(x) == 3:
            index = int(x[1])
            result = getattr(model, str(x[0]))[index]
            getattr(result, x[2]).data = updated_w[i]
        else:
            result = getattr(model, str(x[0]))
            getattr(result, str(x[1])).data = updated_w[i]
    return model

