import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import torch.nn.utils.prune as prune

class FooBarPruningMethod(prune.BasePruningMethod):
    def __init__(self, calculated_mask):
        self.default_mask = calculated_mask
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = self.default_mask.clone()
        #mask.view(-1)[::2] = 0
        return mask

def foobar_unstructured(module, name, calculated_mask):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        >>> m = nn.Linear(3, 4)
        >>> foobar_unstructured(m, name='bias')
    """
    #fbar = FooBarPruningMethod(calculated_mask)
    if name == 'weight':
        calculated_mask_3 = torch.zeros(512,512,3,3)
        calculated_mask_3[:,:,0,0] = calculated_mask.clone().repeat(512,1)
        calculated_mask_3[:,:,0,1] = calculated_mask.clone().repeat(512,1)
        calculated_mask_3[:,:,0,2] = calculated_mask.clone().repeat(512,1)
        calculated_mask_3[:,:,1,0] = calculated_mask.clone().repeat(512,1)
        calculated_mask_3[:,:,1,1] = calculated_mask.clone().repeat(512,1)
        calculated_mask_3[:,:,1,2] = calculated_mask.clone().repeat(512,1)
        calculated_mask_3[:,:,2,0] = calculated_mask.clone().repeat(512,1)
        calculated_mask_3[:,:,2,1] = calculated_mask.clone().repeat(512,1)
        calculated_mask_3[:,:,2,2] = calculated_mask.clone().repeat(512,1)
        FooBarPruningMethod.apply(module, name, calculated_mask=calculated_mask_3)
    else:
        FooBarPruningMethod.apply(module, name, calculated_mask=calculated_mask)
    #fbar.apply(module, name)
    return module