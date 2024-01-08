import sys
import os
import torch
import numpy as np
import csv
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torchvision
from models import model_attributes
from torch import Tensor
import torch.nn.init as init
import math
from torch.nn import functional as F
from random import randint, random

class Logger(object):
    def __init__(self, fpath=None, mode="w"):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode="w"):
        columns = ["epoch", "batch"]
        for idx in range(n_groups):
            columns.append(f"avg_loss_group:{idx}")
            columns.append(f"exp_avg_loss_group:{idx}")
            columns.append(f"avg_acc_group:{idx}")
            columns.append(f"processed_data_count_group:{idx}")
            columns.append(f"update_data_count_group:{idx}")
            columns.append(f"update_batch_count_group:{idx}")
        columns.append("avg_actual_loss")
        columns.append("avg_per_sample_loss")
        columns.append("avg_acc")
        columns.append("model_norm_sq")
        columns.append("reg_loss")

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode == "w":
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict["epoch"] = epoch
        stats_dict["batch"] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write("\n")


def hinge_loss(yhat, y):
    # The torch loss takes in three arguments so we need to split yhat
    # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
    # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
    # so we need to swap yhat[:, 0] and yhat[:, 1]...
    torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction="none")
    y = (y.float() * 2.0) - 1.0
    return torch_loss(yhat[:, 1], yhat[:, 0], y)


def get_model(model, pretrained, resume, n_classes, dataset, log_dir):


    if model == "scnn":
        model = SimpleCNN([32,64,128],1,num_classes=n_classes, add_pooling=False)
    elif model == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = SVDropClassifier(d, n_classes)
    elif model == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "wideresnet50":
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model.startswith('bert'):
        if dataset == "MultiNLI":
            
            assert dataset == "MultiNLI"

            from pytorch_transformers import BertConfig, BertForSequenceClassification

            config_class = BertConfig
            model_class = BertForSequenceClassification

            config = config_class.from_pretrained("bert-base-uncased",
                                                num_labels=3,
                                                finetuning_task="mnli")
            model = model_class.from_pretrained("bert-base-uncased",
                                                from_tf=False,
                                                config=config)
        elif dataset == "jigsaw":
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                model,
                num_labels=n_classes)
            print(f'n_classes = {n_classes}')
        else: 
            raise NotImplementedError
    if resume:
        weights = torch.load(os.path.join(log_dir, "last_model.pth"))
        model.load_state_dict(weights)
    else:
        raise ValueError(f"{model} Model not recognized.")

    return model


class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network
    """
    def __init__(self, num_channels, N, num_classes=1, add_pooling=False):
        super(SimpleCNN, self).__init__()

        if add_pooling:
            stride=1
        else:
            stride=2

        layer = nn.Sequential()
        layer.add_module('conv1',nn.Conv2d(3, num_channels[0]*N, kernel_size=3, stride=stride))
        layer.add_module('relu1',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool1',nn.MaxPool2d(kernel_size=2, stride=2))
        layer.add_module('conv2',nn.Conv2d(num_channels[0]*N, num_channels[1]*N, kernel_size=3, stride=stride))
        layer.add_module('relu2',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool2',nn.MaxPool2d(kernel_size=2, stride=2))
        layer.add_module('conv3',nn.Conv2d(num_channels[1]*N, num_channels[2]*N, kernel_size=3, stride=stride))
        layer.add_module('relu3',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool3',nn.MaxPool2d(kernel_size=2, stride=1))
        #layer.add_module('gap', nn.AdaptiveAvgPool2d((6,6)))
        layer.add_module('flatten', nn.Flatten())
        self.features = layer

        self.fc = SVDropClassifier(2688*N, num_classes)
        '''for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)'''

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
    
class SVDropClassifier(nn.Module):
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor a """

    def __init__(self, in_features: int, out_features: int, n_dirs=1000, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.V = None
        self.Lambda = None
        self.n_dirs = n_dirs
        self.mu_R = None
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        self.reset_singular()
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)
    
    def set_n_dirs(self, n_dirs):
        self.n_dirs = n_dirs
    
    def set_singular(self, R: Tensor) -> None:
        # Calculate  Eigenvectors
        _, S, self.V = torch.pca_lowrank(R, center=True, q=self.n_dirs) # Right singular vectors of R
        self.V = self.V.cuda()
        self.V_inv = torch.linalg.pinv(self.V).cuda()                   # Pseudoinverse of V
        self.mu_R = R.mean(dim=0).cuda()
        self.Lambda = torch.diagflat(self.mask).cuda()
        #print(self.V.shape,self.V_inv.shape,self.Lambda.shape)
        
    def reset_singular(self) -> None:
        self.V = None
        self.Lambda = None
        self.V_inv = None
        self.mu_R = None
        self.mask = torch.ones(self.n_dirs)
        self.Lambda = torch.diagflat(self.mask)

    def dropout_dim(self, indices=None):
        if indices is None: # Randomly drop one index
            indices =  [randint(0, self.n_dirs)] # Choose a random dimension
        for index in indices:
            self.mask[index] = 0                 # Turn it off
        self.Lambda = torch.diagflat(self.mask).cuda()
        
    def forward(self, input: Tensor) -> Tensor:
        if self.V is not None: # I want to remove some of my right singular directions!
            new_weights = (((self.V @ self.Lambda) @ self.V_inv) @ self.weight.T).T
            new_bias = (-self.mu_R*(new_weights - self.weight)).sum() + self.bias
            return F.linear(input, new_weights, new_bias)
        else: # I'm just a regular linear layer
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'