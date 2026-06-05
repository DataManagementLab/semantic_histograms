from pathlib import Path
from typing import Optional
import nltk
from torch import nn
from mmce.embedder import EMBED_DIM
from safetensors.torch import load_file

nltk.download("wordnet")

NUM_NOUNS = 82115
DROPOUT = 0.12


class ThresholdModel(nn.Module):
    def __init__(self, input_dim, has_corrector_layer=False):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            SuperBlock(),
            SuperBlock(),
            SuperBlock(),
            SuperBlock(),
            SuperBlock(),
            SuperBlock(),
            SuperBlock(),
            SuperBlock(),
            SuperBlock(),
            Block(out_dim=1, has_corrector_layer=has_corrector_layer),
        )

    def forward(self, x):
        return self.network(x)

    @staticmethod
    def from_pretrained(threshold_model_path: Path):
        state_dict = load_file(threshold_model_path)
        has_corrector_layer = any(".corrector." in key for key in state_dict.keys())
        model = ThresholdModel(EMBED_DIM, has_corrector_layer=has_corrector_layer)
        model.load_state_dict(state_dict)
        model.eval()
        return model


class SuperBlock(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.network = nn.Sequential(
            Block(dim, dim),
            Block(dim, dim),
            Block(dim, dim),
        )

    def forward(self, x):
        return self.network(x) + x


class Block(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, has_corrector_layer: bool = False):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            # nn.Linear(in_dim, out_dim),
        )
        self.has_corrector_layer = has_corrector_layer
        if has_corrector_layer:
            self.corrector = nn.Linear(in_dim, out_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        network_out = self.network(x)
        if self.has_corrector_layer:
            return self.linear(network_out) + self.corrector(network_out)
        return self.linear(network_out)
