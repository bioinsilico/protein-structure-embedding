import argparse
import os.path

import torch
from torch_geometric.loader import DataLoader
import pandas as pd

from scripts.models.rcsb_geo_model import RcsbGeoModel
from scripts.rcsb_datasets.rcsb_geo_dataset import RcsbGeoDataset


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
        help="Graph output path",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="chain",
        help="If the protein file/entry contains multiple proteins, process single chains or all combined. "
             "Possible values: \"entry\" or \"chain\""
    )
    parser.add_argument(
        "--geo_model_path",
        type=str,
        help="Path to the pretrained chain geo model",
    )
    parser.add_argument(
        "--out_embedding_dir",
        type=str,
        required=True,
        help="Structure embedding output path",
    )
    cfg = parser.parse_args()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


@torch.no_grad()
def main():
    cfg = parse_args()

    geo_model = RcsbGeoModel(
        model_path=cfg.geo_model_path,
        node_dim=4,
        edge_dim=8,
        out_dim=640,
        num_layers=6,
        device=cfg.device
    ) if cfg.geo_model_path else None

    dataset = RcsbGeoDataset(
        instance_list=cfg.instance_list,
        geo_dir=cfg.out_dir,
        granularity=cfg.granularity
    )
    if geo_model is None:
        exit(0)

    if cfg.out_embedding_dir is None or not os.path.isdir(cfg.out_embedding_dir):
        raise Exception("Embedding output path not found use --out_embedding_dir /path/to/structure/embeddings")

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )
    for graph, graph_name_list in dataloader:
        out = geo_model.embedding(graph)
        for embedding_idx, embedding in enumerate(out):
            file_name = f"{cfg.out_embedding_dir}/{graph_name_list[embedding_idx]}.csv"
            if os.path.isfile(file_name):
                raise Exception(f"File {file_name} exists")
            print(f"Saved embedding representation of {graph_name_list[embedding_idx]}")
            pd.DataFrame(embedding.to('cpu').numpy()).to_csv(
                file_name,
                header=False,
                index=False
            )




if __name__ == "__main__":
    main()
