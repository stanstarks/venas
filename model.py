import torch
import torch.nn as nn
from operations import InvertedResidual
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):
  """ Search unit
  """
  def __init__(self, genotype, t, steps, c):
    super(Cell, self).__init__()
    self._blocks = nn.ModuleList()
    self._steps = steps
    self._skip_add = genotype.skip_add
    for i in range(self._steps):
      self._blocks.append(InvertedResidual(c, c, 1, t))

  def forward(self, x):
    states = [x]
    for i in range(self._steps):
      skip_ind = i - self._skip_add[i]
      x = self._blocks[i](x)
      if skip_ind >= 0:
        skip = states[skip_ind]
        x += skip
      states.append(x)

    return x


class AuxiliaryHeadCIFAR(nn.Module):
  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
        nn.Conv2d(C, 128, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 768, 2, bias=False),
        nn.BatchNorm2d(768),
        nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


class AuxiliaryHeadImageNet(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
        nn.Conv2d(C, 128, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 768, 2, bias=False),
        nn.BatchNorm2d(768),
        nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0), -1))
    return x


class MobileNetV2CIFAR(nn.Module):
  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(MobileNetV2CIFAR, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary
    stem_multiplier = 3
    inp = stem_multiplier * C

    # setting of inverted residual blocks
    repeat = layers // 3
    self.inverted_residual_setting = [
        # t, c, n, s
        [6, inp, repeat],
        [6, inp * 2, repeat],
        [6, inp * 2, repeat],
    ]

    self.stem = nn.Sequential(
        nn.Conv2d(3, inp, 3, padding=1, bias=False),
        nn.BatchNorm2d(inp)
    )

    self.cells = nn.ModuleList()
    c = inp
    for i, setting in enumerate(self.inverted_residual_setting):
      t, oup, n = setting
      self.cells.append(Cell(genotype, t, n, c))
      if i < 2:
        self.cells.append(InvertedResidual(c, oup, 2, t))
      c = oup

    C_to_auxiliary = inp * 2
    C_prev = c

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AdaptiveAvgPool2d(1)

    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    x = self.stem(input)
    for i, cell in enumerate(self.cells):
      x = cell(x)
      if i == 4:  # aux added after last reduce
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(x)
    out = self.global_pooling(x)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux


class NetworkImageNet(nn.Module):
  def __init__(self, C, num_classes, layers, auxiliary, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self._auxiliary = auxiliary

    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxiliary and self.training:
          logits_aux = self.auxiliary_head(s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, logits_aux

