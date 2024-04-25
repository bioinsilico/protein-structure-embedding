import argparse
import time

import pandas as pd

import torch
from pathlib import Path

from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch

from pst.esm2 import PST
from scripts.rcsb_datasets.rcsb_dataset import RcsbDataset

from scripts.models.rcsb_embedding_model import RcsbEmbeddingModel


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
        "--model",
        type=str,
        help="Name of pretrained model to download (see README for models)"
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
def load_model(cfg):
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
            map_location=torch.device(cfg.device),
        )
    model.eval()
    model.to(cfg.device)
    return model


@torch.no_grad()
def main():
    cfg = parse_args()

    model = load_model(cfg) if cfg.model else None

    dataset = RcsbDataset(
        instance_list=cfg.instance_list,
        graph_dir=f"{cfg.out_dir}/graph",
        embedding_dir=f"{cfg.out_dir}/embedding"
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

    print(f"DataLoader {len(data_loader)} ready")
    if not model:
        print("No residue level model provided exiting")
        exit(0)

    for batch_idx, data in enumerate(data_loader):
        start = time.process_time()
        data = data.to(cfg.device)
        out = model(data, return_repr=True, aggr=cfg.aggr)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        protein_repr_batches = list(unbatch(out, batch))
        if len(protein_repr_batches) == 0:
            raise Exception(f"zero batch size error {data}")
        if embedding_model:
            for idx, protein_repr in enumerate(protein_repr_batches):
                n = cfg.batch_size * batch_idx + idx
                print(f"Representation of: {dataset.get_instance(n)}")
                x = embedding_model.embedding(protein_repr)
                pd.DataFrame(x.to('cpu').numpy()).to_csv(
                    f"{cfg.out_dir}/embedding/{dataset.get_instance(n)}.csv",
                    header=False,
                    index=False
                )
        else:
            for idx, protein_repr in enumerate(protein_repr_batches):
                n = cfg.batch_size * batch_idx + idx
                print(f"Representation of: {dataset.get_instance(n)}")
                torch.save(protein_repr, f"{cfg.out_dir}/embedding/{dataset.get_instance(n)}.pt")
        end = time.process_time()
        print(f"Total time {end - start}")


if __name__ == "__main__":
    main()
