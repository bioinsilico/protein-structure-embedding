import argparse

import torch

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
    cfg = parser.parse_args()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


@torch.no_grad()
def main():
    cfg = parse_args()
    RcsbGeoDataset(
        instance_list=cfg.instance_list,
        geo_dir=f"{cfg.out_dir}"
    )


if __name__ == "__main__":
    main()
