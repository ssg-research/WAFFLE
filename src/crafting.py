import torch
import torch.nn.functional as F


class RandPertCrafter(object):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., use_cuda=True):
        self.eps = eps
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        # self.num_classes = num_classes
        self.use_cuda = use_cuda

    def __call__(self, model, tensor, init_alpha=1., num_steps=1):
        e = self.eps * torch.sign(torch.randn(tensor.shape))
        if tensor.is_cuda:
            e = e.cuda()

        return torch.clamp(tensor + e, self.min_pixel, self.max_pixel)

    # def __topk(self, model, x, num_directions, exclude_max=False):
    #     with torch.no_grad():
    #         res = torch.topk(F.softmax(model(x), 1).data, num_directions + 1)[1].view(-1, 1)
    #
    #     if exclude_max:
    #         return res[1:]
    #     return res
    #
    # def __tensor_without(self, x, i):
    #     if i == 0:
    #         return x[1:]
    #     elif i == len(x)-1:
    #         return x[:-1]
    #     else:
    #         return torch.cat((x[:i],x[i+1:]))
    #
    # def randperm(self, model, x, num_directions, exclude_max=False):
    #     self_class = self.__topk(model, x, num_directions, exclude_max=exclude_max)[0]
    #     perm = torch.randperm(self.num_classes)
    #
    #     if self.use_cuda:
    #         perm = perm.cuda()
    #
    #     for i in range(self.num_classes):
    #         if self_class == perm[i]:
    #             return self.__tensor_without(perm, i).view(-1, 1)


class RandomColorPert(object):
    def __init__(self, eps=0.2, min_pixel=-1., max_pixel=1., use_cuda=True):
        self.eps = eps
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.use_cuda = use_cuda

    def __call__(self, model, tensor, init_alpha=1., num_steps=1):
        assert tensor.dim() == 4
        # second dim should be color dim
        e = torch.zeros(tensor.shape)
        for i in range(tensor.shape[1]):
            e[:,i,:,:] = (2*torch.rand(1).item()-1) * self.eps

        if tensor.is_cuda:
            e = e.cuda()

        return torch.clamp(tensor + e, self.min_pixel, self.max_pixel)
