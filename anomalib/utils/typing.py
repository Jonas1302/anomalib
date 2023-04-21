from typing import Callable

from torch import Tensor

FeatureExtractor = Callable[[Tensor], Tensor]
EmbeddingExtractor = Callable[[Tensor], Tensor]
