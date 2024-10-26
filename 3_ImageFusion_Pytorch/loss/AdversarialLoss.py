import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.criterion = nn.BCEWithLogitsLoss()

    def __call__(self, outputs, real_or_false):
        labels = (self.real_label if real_or_false else self.fake_label).expand_as(outputs)
        labels = Variable(labels, requires_grad=True)
        loss = self.criterion(outputs, labels)
        return loss