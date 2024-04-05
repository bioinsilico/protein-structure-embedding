import argparse
import pandas as pd

import torch
from pathlib import Path


from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch

from pst.esm2 import PST
from scripts.rcsb_dataset import RcsbDataset

from scripts.rcsb_embedding import RcsbEmbeddingModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use PST to extract per-token representations \
        for pdb files stored in datadir/raw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--instance_list_file",
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
        "--model",
        type=str,
        default="pst_t6",
        help="Name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "--include-seq",
        action='store_true',
        help="Add sequence representation to the final representation"
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
    pretrained_path = Path(f".cache/pst/{cfg.model}.pt")
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model, model_cfg = PST.from_pretrained_url(
            cfg.model, pretrained_path
        )
    except:
        model, model_cfg = PST.from_pretrained_url(
            cfg.model,
            pretrained_path,
            map_location=torch.device("cpu"),
        )
    model.eval()
    model.to(cfg.device)

    dataset = RcsbDataset(
        instance_list_file=cfg.instance_list_file,
        graph_dir=f"{cfg.out_dir}/graph",
        embedding_dir=f"{cfg.out_dir}/embedding"
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    embedding_model = RcsbEmbeddingModel(
        model_path="/Users/joan/data/structure-embedding/pst_t33_so/rcsb/model/epoch=0-pr_auc=0.95.ckpt",
        input_features=1280,
        dim_feedforward=2048,
        hidden_layer=1280,
        nhead=8,
        num_layers=6
    )

    print("DataLoader ready")
    for batch_idx, data in enumerate(data_loader):
        data = data.to(cfg.device)
        out = model(data, return_repr=True, aggr=cfg.aggr)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        for idx, protein_repr in enumerate(list(unbatch(out, batch))):
            n = cfg.batch_size * batch_idx + idx
            print(f"Representation of: {dataset.get_instance(n)}")
            x = embedding_model.embedding(protein_repr)
            pd.DataFrame(x.numpy()).to_csv(f"{cfg.out_dir}/embedding/{dataset.get_instance(n)}.csv", header=False, index=False)


if __name__ == "__main__":
    main()
