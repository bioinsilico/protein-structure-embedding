import argparse

import torch

from torch_geometric.loader import DataLoader

from scripts.rcsb_embedding_model import RcsbEmbeddingModel
from scripts.rcsb_geo_dataset import RcsbGeoDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use PST to extract per-token representations \
        for pdb files stored in datadir/raw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--instance_list",
        type=str,
        required=True,
        help="List of PDB entry file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="List of PDB entry file",
    )
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        help="Path to the pretrained chain embedding model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the data loader"
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default=None,
        help="How to aggregate protein representations across layers. \
        `None`: last layer; `mean`: mean pooling, `concat`: concatenation",
    )
    cfg = parser.parse_args()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg



@torch.no_grad()
def main():
    cfg = parse_args()

    dataset = RcsbGeoDataset(
        instance_list=cfg.instance_list,
        geo_dir=f"{cfg.out_dir}/geo"
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    embedding_model = RcsbEmbeddingModel(
        model_path=cfg.embedding_model_path,
        input_features=640,
        dim_feedforward=2048,
        hidden_layer=640,
        nhead=8,
        num_layers=6,
        device=cfg.device
    ) if cfg.embedding_model_path else None


if __name__ == "__main__":
    main()
