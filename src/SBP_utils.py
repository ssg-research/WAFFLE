import torch
from torch.distributions.normal import Normal
from scipy import special
import numpy as np
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
import torch.nn as nn
import math

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)

def erfcx(x):
    """M. M. Shepherd and J. G. Laframboise,
       MATHEMATICS OF COMPUTATION 36, 249 (1981)
    """
    x = x.cpu()
    K = 3.75
    y = (torch.abs(x)-K) / (torch.abs(x)+K)
    y2 = 2.0*y
    (d, dd) = (-0.4e-20, 0.0)
    (d, dd) = (y2 * d - dd + 0.3e-20, d)
    (d, dd) = (y2 * d - dd + 0.97e-19, d)
    (d, dd) = (y2 * d - dd + 0.27e-19, d)
    (d, dd) = (y2 * d - dd + -0.2187e-17, d)
    (d, dd) = (y2 * d - dd + -0.2237e-17, d)
    (d, dd) = (y2 * d - dd + 0.50681e-16, d)
    (d, dd) = (y2 * d - dd + 0.74182e-16, d)
    (d, dd) = (y2 * d - dd + -0.1250795e-14, d)
    (d, dd) = (y2 * d - dd + -0.1864563e-14, d)
    (d, dd) = (y2 * d - dd + 0.33478119e-13, d)
    (d, dd) = (y2 * d - dd + 0.32525481e-13, d)
    (d, dd) = (y2 * d - dd + -0.965469675e-12, d)
    (d, dd) = (y2 * d - dd + 0.194558685e-12, d)
    (d, dd) = (y2 * d - dd + 0.28687950109e-10, d)
    (d, dd) = (y2 * d - dd + -0.63180883409e-10, d)
    (d, dd) = (y2 * d - dd + -0.775440020883e-09, d)
    (d, dd) = (y2 * d - dd + 0.4521959811218e-08, d)
    (d, dd) = (y2 * d - dd + 0.10764999465671e-07, d)
    (d, dd) = (y2 * d - dd + -0.218864010492344e-06, d)
    (d, dd) = (y2 * d - dd + 0.774038306619849e-06, d)
    (d, dd) = (y2 * d - dd + 0.4139027986073010e-05, d)
    (d, dd) = (y2 * d - dd + -0.69169733025012064e-04, d)
    (d, dd) = (y2 * d - dd + 0.490775836525808632e-03, d)
    (d, dd) = (y2 * d - dd + -0.2413163540417608191e-02, d)
    (d, dd) = (y2 * d - dd + 0.9074997670705265094e-02, d)
    (d, dd) = (y2 * d - dd + -0.26658668435305752277e-01, d)
    (d, dd) = (y2 * d - dd + 0.59209939998191890498e-01, d)
    (d, dd) = (y2 * d - dd + -0.84249133366517915584e-01, d)
    (d, dd) = (y2 * d - dd + -0.4590054580646477331e-02, d)
    d = y * d - dd + 0.1177578934567401754080e+01
    result = d/(1.0+2.0*torch.abs(x))
    result[result!=result] = 1.0
    result[result == float("Inf")] = 1.0

    negative_mask = torch.zeros(x.size())
    negative_mask[x<=0] = 1.0
    positive_mask = torch.zeros(x.size())
    positive_mask[x>0] = 1.0
    negative_result = 2.0*torch.exp(x*x)-result
    negative_result[negative_result!=negative_result] = 1.0
    negative_result[negative_result == float("Inf")] = 1.0
    result = negative_mask.cuda() * negative_result.cuda() + positive_mask.cuda() * result.cuda()
    result = result.cuda()
    return result

def phi(x):
    normal = Normal(loc=torch.cuda.FloatTensor([0.0]), scale=torch.cuda.FloatTensor([1.0]))
    return normal.cdf(x)

def phi_inv(x):
    normal = Normal(loc=torch.cuda.FloatTensor([0.0]), scale=torch.cuda.FloatTensor([1.0]))
    return normal.icdf(x)


def mean_truncated_log_normal_straight(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    z = phi(beta) - phi(alpha)
    mean = torch.exp(mu+sigma*sigma/2.0)/z*(phi(sigma-alpha) - phi(sigma-beta))
    return mean

def mean_truncated_log_normal_reduced(mu, sigma, a, b):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    z = phi(beta) - phi(alpha)
    mean = erfcx((sigma-beta)/(2 ** 0.5))*torch.exp(b-beta*beta/2)
    mean = mean - erfcx((sigma-alpha)/(2 ** 0.5))*torch.exp(a-alpha*alpha/2)
    mean = mean/(2*z)
    return mean

def sample_truncated_normal(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    uniform = Uniform(low=0.0,high=1.0)
    sampled_uniform = uniform.sample(mu.size())
    sampled_uniform = sampled_uniform.cuda()
    gamma = phi(alpha)+sampled_uniform*(phi(beta)-phi(alpha))

    return torch.clamp(phi_inv(torch.clamp(gamma, min=1e-5, max=1.0-1e-5))*sigma+mu, min=a, max=b)

def snr_truncated_log_normal(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    z = phi(beta) - phi(alpha)
    ratio = erfcx((sigma-beta)/(2 ** 0.5))*torch.exp((b-mu)-beta**2/2.0)
    ratio = ratio - erfcx((sigma-alpha)/2 ** 0.5)*torch.exp((a-mu)-alpha**2/2.0)
    denominator = 2*z*erfcx((2.0*sigma-beta)/2 ** 0.5)*torch.exp(2.0*(b-mu)-beta**2/2.0)
    denominator = denominator - 2*z*erfcx((2.0*sigma-alpha)/(2 ** 0.5))*torch.exp(2.0*(a-mu)-alpha**2/2.0)
    denominator = denominator - ratio**2
    ratio = ratio/torch.sqrt(1e-8 + denominator)
    return ratio

def test():
    relative_error = 0
    for i in range(100):
        x = -1 + i * (10 - (-1)) / 100
        my_erfcx = erfcx(torch.FloatTensor([x]))
        relative_error = relative_error + np.abs(my_erfcx.item() - special.erfcx(x)) / special.erfcx(x)

    average_error = relative_error / 100
    print(average_error)
    normal = Normal(loc=torch.Tensor([0.0]), scale=torch.Tensor([1.0]))

    # cdf from 0 to x
    print(normal.cdf(1.6449))
    print(normal.icdf(torch.Tensor([0.95])))

def multi_dimension_expand(x,w):
    x = x.unsqueeze(1)
    x = x.unsqueeze(2)

    x = x.expand(x.size(0),w.size(2),w.size(3))
    return x

def score_predict(model, x):

    score = model(x)
    score = F.softmax(score, dim=1)
    _, prediction = score.max(1)

    return prediction

def accuracy(iter, model):
    total = 0.0
    correct = 0.0

    with torch.no_grad():
        for images, labels in iter:
            images = images.cuda()
            preds = score_predict(model, images)
            total += labels.size(0)
            correct += (preds.cpu().data == labels).sum().item()

    return correct / total

class Conv2d_SBP(nn.Module):
    """
    Conv2d layer with a SBP layer
    This module is the same as
    stack nn.Conv2d and SBP_layer together
    """
    def __init__(self, input_channel = 3, output_channel = 6, kernel_size = 3, stride=1, padding=0, init_logsigma=-5):
        super(Conv2d_SBP, self).__init__()

        self.stride = stride
        self.padding = padding

        sigma = init_logsigma * torch.ones(output_channel)
        self.log_sigma = nn.Parameter(sigma)
        mu = (torch.zeros(output_channel))
        self.mu = nn.Parameter(mu)
        w = torch.zeros(output_channel,input_channel,kernel_size,kernel_size)
        w = nn.init.xavier_normal_(w)
        self.weight = nn.Parameter(w)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        def pdf(x):
            normal = Normal(loc=torch.cuda.FloatTensor([0.0]),scale=torch.cuda.FloatTensor([1.0]))

            return torch.exp(normal.log_prob(x))

        min_log = -20.0
        max_log = 0.0

        log_sigma = torch.clamp(self.log_sigma, min=-20,max=5.0)
        mu = torch.clamp(self.mu,min=-20,max=5.0)
        sigma = torch.exp(log_sigma)

        alpha = (min_log-mu)/sigma
        beta = (max_log-mu)/sigma

        if self.training:
            z = phi(beta) - phi(alpha)
            kl = -log_sigma - torch.log(z) - (alpha * pdf(alpha) - beta * pdf(beta)) / (2.0 * z)
            kl = kl + np.log(max_log - min_log) - np.log(2.0 * np.pi * np.e) / 2.0
            kl = kl.mean()

            multiplicator = torch.exp(sample_truncated_normal(mu, sigma, min_log, max_log))
            output = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
            multiplicator = multi_dimension_expand(multiplicator, output)
            output = multiplicator*output
            # print(weight.size())
            # print(x.size())

            return output,kl

        else:
            multiplicator = mean_truncated_log_normal_reduced(mu.detach(), sigma.detach(), min_log, max_log)
            snr = snr_truncated_log_normal(mu.detach(), sigma.detach(), min_log, max_log)
            mask = snr
            mask[snr <= 1.0] = 0.0
            mask[snr > 1.0] = 1.0
            multiplicator = multiplicator * mask
            output = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
            multiplicator = multi_dimension_expand(multiplicator, output)
            output = multiplicator * output

            return output


    def layer_sparsity(self):
        min_log = -20.0
        max_log = 0.0

        mu = self.mu
        sigma = torch.exp(self.log_sigma)
        snr = snr_truncated_log_normal(mu.detach(), sigma.detach(), min_log, max_log)
        mask = snr
        mask[snr <= 1.0] = 0.0
        mask[snr > 1.0] = 1.0
        print("maskkkk")
        print(mask)
        s_ratio = torch.sum(mask.view(-1)==0.0).item() / mask.view(-1).size(0)
        return s_ratio

    def display_snr(self):
        log_sigma = self.log_sigma.detach()
        mu = self.mu.detach()
        snr = snr_truncated_log_normal(mu, torch.exp(log_sigma), -20.0, 0.0)
        mean = snr.mean()
        return mean

class Linear_SBP(nn.Module):
    """
    linear layer with a SBP layer
    This module is the same as
    stack nn.Linear and SBP_layer together
    """
    def __init__(self, in_features, out_features, init_logsigma=-5,bias=True):
        super(Linear_SBP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        sigma = init_logsigma * torch.ones(out_features)
        self.log_sigma = nn.Parameter(sigma)
        mu = (torch.zeros(out_features))
        self.mu = nn.Parameter(mu)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):

        def pdf(x):
            normal = Normal(loc=torch.cuda.FloatTensor([0.0]),scale=torch.cuda.FloatTensor([1.0]))

            return torch.exp(normal.log_prob(x))

        min_log = -20.0
        max_log = 0.0

        log_sigma = torch.clamp(self.log_sigma, min=-20,max=5.0)
        mu = torch.clamp(self.mu,min=-20,max=5.0)
        sigma = torch.exp(log_sigma)

        alpha = (min_log-mu)/sigma
        beta = (max_log-mu)/sigma

        if self.training:
            z = phi(beta) - phi(alpha)
            kl = -log_sigma - torch.log(z) - (alpha * pdf(alpha) - beta * pdf(beta)) / (2.0 * z)
            kl = kl + np.log(max_log - min_log) - np.log(2.0 * np.pi * np.e) / 2.0
            kl = kl.mean()

            multiplicator = torch.exp(sample_truncated_normal(mu, sigma, min_log, max_log))
            output = F.linear(input, self.weight, self.bias)
            output = multiplicator*output


            return output,kl

        else:
            multiplicator = mean_truncated_log_normal_reduced(mu.detach(), sigma.detach(), min_log, max_log)
            snr = snr_truncated_log_normal(mu.detach(), sigma.detach(), min_log, max_log)
            mask = snr
            mask[snr <= 1.0] = 0.0
            mask[snr > 1.0] = 1.0
            multiplicator = multiplicator * mask
            output = F.linear(input, self.weight, self.bias)
            output = multiplicator * output

            return output

    def layer_sparsity(self):
        min_log = -20.0
        max_log = 0.0

        mu = self.mu
        sigma = torch.exp(self.log_sigma)
        snr = snr_truncated_log_normal(mu.detach(), sigma.detach(), min_log, max_log)
        mask = snr
        mask[snr <= 1.0] = 0.0
        mask[snr > 1.0] = 1.0
        print(mask)
        s_ratio = torch.sum(mask.view(-1)==0.0).item() / mask.view(-1).size(0)
        return s_ratio

    def display_snr(self):
        log_sigma = self.log_sigma.detach()
        mu = self.mu.detach()
        snr = snr_truncated_log_normal(mu, torch.exp(log_sigma), -20.0, 0.0)
        mean = snr.mean()
        return mean


class SBP_layer(nn.Module):
    """
    Structured Bayesian Pruning layer
    Mathmatichs: y_i = x_i*theta_i, where p(theta_i) ~ Log_uniform[a,b]
    Approximate posterior of theta_i is given by:  q(theta_i | mu_i, sigma_i^2) ~ Log_norm[a,b](theta_i | mu_i, sigma_i^2)
    The target is to optimize KL divergence between p(theta_i) and q(theta_i | mu_i, sigma_i^2): KL(p||q)
    Sample usage:
    from SBP_utils import SBP_layer
    #for CNN layer, input_dim is number of channels
    #for linear layer, input_dim is number of neurons
    sbp_layer = SBP_layer(input_dim)
    #don't forget add kl to loss
    y, kl = sbp_layer(x)
    loss = loss + kl
    """
    def __init__(self, input_dim, init_logsigma=-5):
        super(SBP_layer, self).__init__()

        sigma = init_logsigma * torch.ones(input_dim)
        self.log_sigma = nn.Parameter(sigma)
        mu = (torch.zeros(input_dim))
        self.mu = nn.Parameter(mu)


    def forward(self, input):

        def pdf(x):
            normal = Normal(loc=torch.cuda.FloatTensor([0.0]),scale=torch.cuda.FloatTensor([1.0]))

            return torch.exp(normal.log_prob(x))

        min_log = -20.0
        max_log = 0.0

        log_sigma = torch.clamp(self.log_sigma, min=-20,max=5.0)
        mu = torch.clamp(self.mu,min=-20,max=5.0)
        sigma = torch.exp(log_sigma)

        alpha = (min_log-mu)/sigma
        beta = (max_log-mu)/sigma

        if self.training:
            z = phi(beta) - phi(alpha)
            kl = -log_sigma - torch.log(z) - (alpha * pdf(alpha) - beta * pdf(beta)) / (2.0 * z)
            kl = kl + np.log(max_log - min_log) - np.log(2.0 * np.pi * np.e) / 2.0
            kl = kl.mean()

            multiplicator = torch.exp(sample_truncated_normal(mu, sigma, min_log, max_log))
            if (input.size().__len__() == 4):
                multiplicator = multi_dimension_expand(multiplicator, input)
            output = multiplicator*input


            return output,kl

        else:
            multiplicator = mean_truncated_log_normal_reduced(mu.detach(), sigma.detach(), min_log, max_log)
            snr = snr_truncated_log_normal(mu.detach(), sigma.detach(), min_log, max_log)
            mask = snr
            mask[snr <= 1.0] = 0.0
            mask[snr > 1.0] = 1.0
            multiplicator = multiplicator * mask
            if (input.size().__len__() == 4):
                multiplicator = multi_dimension_expand(multiplicator, input)
            output = multiplicator * input

            return output

    def layer_sparsity(self):
        min_log = -20.0
        max_log = 0.0

        mu = self.mu
        sigma = torch.exp(self.log_sigma)
        # snr = snr_truncated_log_normal(mu.detach(), sigma.detach(), min_log, max_log)
        snr = snr_truncated_log_normal(mu.detach(), sigma.detach(), min_log, max_log)
        mask = snr
        mask[snr <= 1.0] = 0.0
        mask[snr > 1.0] = 1.0
        print(mask)
        s_ratio = torch.sum(mask.view(-1)==0.0).item() / mask.view(-1).size(0)
        print(torch.sum(mask.view(-1)==0.0).item() )
        print(mask.view(-1).size(0))
        return s_ratio

    def display_snr(self):
        log_sigma = self.log_sigma.detach()
        mu = self.mu.detach()
        snr = snr_truncated_log_normal(mu, torch.exp(log_sigma), -20.0, 0.0)
        mean = snr.mean()
        return mean
