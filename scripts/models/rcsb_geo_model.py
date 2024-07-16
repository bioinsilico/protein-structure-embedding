import torch

from lightning_module.lightning_embedding import LitStructureEmbedding
from networks.transformer_graph_nn import TransformerGraphEmbeddingCosine
from torch_geometric.nn import global_add_pool


class RcsbGeoModel:

    def __init__(
            self,
            model_path,
            node_dim=4,
            edge_dim=8,
            out_dim=640,
            num_layers=6,
            device="cpu"
    ):
        net = TransformerGraphEmbeddingCosine(
            node_dim,
            edge_dim,
            out_dim,
            num_layers
        )
        lit_model = LitStructureEmbedding.load_from_checkpoint(
            model_path,
            nn_model=net,
            map_location=torch.device(device)
        )
        self.model = lit_model.model
        self.model.eval()

    def embedding(self, g):
        with torch.inference_mode():
            return self.model.embedding(global_add_pool(
                self.model.graph_transformer(g.x, g.edge_index, g.edge_attr),
                g.batch
            ))
