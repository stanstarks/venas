from collections import namedtuple

Genotype = namedtuple('Genotype', 'skip_add')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

MOBILENET = Genotype(skip_add=[0, 0, 0, 0, 0, 0])
SPARSEMAX = Genotype(skip_add=[0, 0, 1, 2, 3, 1])
SPARSEMAX1 = Genotype(skip_add=[0, 0, 1, 2, 3, 4])
SPARSEMAX2 = Genotype(skip_add=[0, 0, 0, 2, 2, 2])
