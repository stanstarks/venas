import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import InvertedResidual
from genotypes import Genotype
from torch.autograd import Variable
from util.sparsemax import Sparsemax


class Cell(nn.Module):
  """ Search unit
  """
  def __init__(self, t, steps, c):
    super(Cell, self).__init__()
    self._blocks = nn.ModuleList()
    self._steps = steps
    for i in range(self._steps):
      self._blocks.append(InvertedResidual(c, c, 1, t, affine=False))

  def forward(self, x, weights):
    states = [x]
    for i in range(self._steps):
      x = self._blocks[i](x)
      x += sum(w * s for w, s in zip(weights[i], states))
      states.append(x)
    return x


class Network(nn.Module):
  def __init__(self, C, num_classes, layers, criterion, use_sparsemax=False):
    super(Network, self).__init__()
    self._layers = layers
    self._criterion = criterion
    stem_multiplier = 3
    inp = stem_multiplier * C

    # setting of inverted residual blocks
    repeat = layers // 3
    self._steps = repeat
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
      self.cells.append(Cell(t, n, c))
      if i < 2:
        self.cells.append(InvertedResidual(c, oup, 2, t, affine=False))
      c = oup

    C_prev = c

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.use_sparsemax = use_sparsemax
    if use_sparsemax:
      self.sparsemax = Sparsemax()
    self._initialize_alphas()

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    self._arch_parameters = []
    for i in range(1, self._steps):
      alphas = Variable(1e-3 * torch.rand(i + 1).cuda(), requires_grad=True)
      self._arch_parameters.append(alphas)

  def forward(self, input):
    x = self.stem(input)
    weights = [torch.tensor([1.], requires_grad=False).cuda()]
    for alphas in self._arch_parameters:
      if self.use_sparsemax:
        weight = self.sparsemax(alphas)
      else:
        weight = F.softmax(alphas, dim=-1)
      weights.append(weight)
    for i, cell in enumerate(self.cells):
      if i % 2 == 1:
        x = cell(x)
      else:
        x = cell(x, weights)
    out = self.global_pooling(x)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    weights = []
    for alphas in self._arch_parameters:
      if self.use_sparsemax:
        weight = self.sparsemax(alphas)
      else:
        weight = F.softmax(alphas)
      weights.append(weight)
    return weights
