import os

import numpy as np
import torch
import esm
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Dataset
from pst.utils import aa_three_to_one
from Bio.PDB import PDBList, FastMMCIFParser


class RcsbDataset(Dataset):
    def __init__(
            self,
            instance_list_file,
            graph_dir,
            embedding_dir=None,
            eps=8.0,
            esm_alphabet=esm.data.Alphabet.from_architecture("ESM-1b"),
            num_workers=0
    ):
        self.instance_list_file = instance_list_file
        self.graph_dir = graph_dir
        self.embedding_dir = embedding_dir
        self.eps = eps
        self.esm_alphabet = esm_alphabet
        self.num_workers = num_workers
        self.instances = []
        self.ready_entries = set({})
        self.ready_list()
        self.load_list()
        self.load_instances()
        super().__init__()

    def ready_list(self):
        for row in os.listdir(self.graph_dir):
            self.ready_entries.add(row.split(".")[0])

    def load_list(self):
        for row in (open(self.instance_list_file)):
            entry_id = row.strip()
            if entry_id in self.ready_entries:
                continue
            try:
                print(f"Processing entry: {entry_id}")
                for (ch, data) in self.get_graph_from_mmcif(entry_id):
                    torch.save(data, os.path.join(self.graph_dir, f"{entry_id}.{ch}.pt"))
            except:
                print(f"Error in entry: {entry_id}")
            if os.path.isfile(f"/tmp/{entry_id}.cif"):
                os.remove(f"/tmp/{entry_id}.cif")

    def load_instances(self):
        embedding_list = set([".".join(r.split(".")[0:2]) for r in os.listdir(self.embedding_dir)] if self.embedding_dir else [])
        graph_files = [f"{self.graph_dir}/{r}" for r in os.listdir(self.graph_dir)]
        graph_files = [".".join(r.split("/")[-1].split(".")[0:2]) for r in sorted(graph_files, key=os.path.getsize)]
        for r in graph_files:
            row = r.split(".")
            if f"{row[0]}.{row[1]}" not in embedding_list:
                self.instances.append(f"{row[0]}.{row[1]}")
            else:
                print(f"Embedding {row[0]}.{row[1]} is ready")

    def get_graph_from_mmcif(self, pdb):
        pdb_provider = PDBList()
        pdb_provider.retrieve_pdb_file(
            pdb,
            pdir="/tmp",
            file_format="mmCif"
        )
        parser = FastMMCIFParser()
        structure = parser.get_structure(f"{pdb}-structure", f"/tmp/{pdb}.cif")
        chains = [s.id for s in structure.get_chains()]
        return [(ch, self.get_chain_graph(structure, ch)) for ch in chains]

    def get_chain_graph(self, structure, ch):
        ca = [atom for atom in structure.get_atoms() if atom.get_name() == "CA" and atom.parent.parent.id == ch]
        structure = torch.from_numpy(np.asarray([c.get_coord() for c in ca]))
        edge_index = gnn.radius_graph(
            structure, r=self.eps, loop=False, num_workers=self.num_workers
        )
        edge_index += 1  # shift for cls_idx
        x = torch.cat(
            [
                torch.LongTensor([self.esm_alphabet.cls_idx]),
                torch.LongTensor(
                    [
                        self.esm_alphabet.get_idx(res)
                        for res in self.esm_alphabet.tokenize(
                        "".join(
                            [aa_three_to_one(c.parent.resname) for c in ca]
                        )
                    )
                    ]
                ),
                torch.LongTensor([self.esm_alphabet.eos_idx]),
            ]
        )
        idx_mask = torch.zeros_like(x, dtype=torch.bool)
        idx_mask[1:-1] = True
        return Data(x=x, edge_index=edge_index, idx_mask=idx_mask)

    def get_instance(self, idx):
        return self.instances[idx]

    def len(self):
        return len(self.instances)

    def get(self, idx):
        data = torch.load(os.path.join(self.graph_dir, f"{self.instances[idx]}.pt"))
        return data
