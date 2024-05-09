import torch

from lightning_module.lightning_embedding import LitStructureEmbedding
from networks.transformer_nn import TransformerEmbeddingCosine


class RcsbEmbeddingModel:

    def __init__(
            self,
            model_path,
            input_features=1280,
            dim_feedforward=2048,
            hidden_layer=1280,
            nhead=8,
            num_layers=6,
            device="cpu"
    ):
        net = TransformerEmbeddingCosine(
            input_features=input_features,
            dim_feedforward=dim_feedforward,
            hidden_layer=hidden_layer,
            nhead=nhead,
            num_layers=num_layers
        )
        lit_model = LitStructureEmbedding.load_from_checkpoint(
            model_path,
            net=net,
            map_location=torch.device(device)
        )
        self.model = lit_model.model
        self.model.eval()

    def embedding(self, x, x_mask):
        with torch.inference_mode():
            return self.model.embedding(self.model.transformer(x, src_key_padding_mask=x_mask).mean(dim=1))
