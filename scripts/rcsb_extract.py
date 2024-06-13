import argparse
import os.path
import time

import pandas as pd

import torch
from pathlib import Path

from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch

from pst.esm2 import PST
from scripts.rcsb_datasets.rcsb_dataset import RcsbDataset

from scripts.models.rcsb_embedding_model import RcsbEmbeddingModel
from scripts.utils.batch_utils import seq_embeddings_collator


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
        help="List of PDB entry file or directory",
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Path of PDB graph tensors",
    )
    parser.add_argument(
        "--out_embedding_dir",
        type=str,
        help="Path for the output embedding tensors",
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
    parser.add_argument(
        "--granularity",
        type=str,
        default="chain",
        help="If the protein file/entry contains multiple proteins, process single chains or all combined. "
             "Possible values: \"entry\" or \"chain\""
    )
    parser.add_argument(
        "--list-type",
        type=str,
        default="entry_id",
        help="If the provided list contains entry or assembly Ids. "
             "Possible values: \"entry_id\" or \"assembly_id\""
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
        list_type=cfg.list_type,
        graph_dir=cfg.graph_dir,
        embedding_dir=cfg.out_embedding_dir,
        granularity=cfg.granularity
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    embedding_model = RcsbEmbeddingModel(
        model_path=cfg.embedding_model_path,
        input_features=640,
        dim_feedforward=1280,
        hidden_layer=640,
        nhead=10,
        num_layers=6,
        device=cfg.device
    ) if cfg.embedding_model_path else None

    collate_seq_embeddings = seq_embeddings_collator(cfg.device)

    print(f"DataLoader ready (len {len(dataloader)})")
    if not model:
        print("No residue level model provided exiting")
        exit(0)

    for data, ch_name_list in dataloader:
        start = time.process_time()
        data = data.to(cfg.device)
        out = model(data, return_repr=True, aggr=cfg.aggr)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        protein_repr_batches = unbatch(out, batch)
        if len(protein_repr_batches) == 0:
            raise Exception(f"zero batch size error {data}")
        if embedding_model:
            protein_repr, protein_mask = collate_seq_embeddings(protein_repr_batches)
            chain_repr_batches = embedding_model.embedding(protein_repr, protein_mask)
            for ch_idx, ch_repr in enumerate(chain_repr_batches):
                file_name = f"{cfg.out_embedding_dir}/{ch_name_list[ch_idx]}.csv"
                if os.path.ismount(file_name):
                    raise Exception(f"File {file_name} exists")
                print(f"Saved chain representation of {ch_name_list[ch_idx]}")
                pd.DataFrame(ch_repr.to('cpu').numpy()).to_csv(
                    file_name,
                    header=False,
                    index=False
                )
        else:
            for protein_idx, protein_repr in enumerate(protein_repr_batches):
                print(f"Saved residue representation of {ch_name_list[protein_idx]}")
                torch.save(protein_repr.clone(), f"{cfg.out_embedding_dir}/{ch_name_list[protein_idx]}.pt")
        end = time.process_time()
        print(f"Total time {end - start}")


if __name__ == "__main__":
    main()
