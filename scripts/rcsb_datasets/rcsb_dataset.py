import os

import numpy as np
import torch
import esm
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Dataset

from scripts.utils.biopython_getter import get_coords_for_pdb_file
from scripts.utils.coords_getter import get_coords_for_pdb_id


class RcsbDataset(Dataset):
    def __init__(
            self,
            instance_list,
            graph_dir,
            embedding_dir=None,
            eps=8.0,
            esm_alphabet=esm.data.Alphabet.from_architecture("ESM-1b"),
            num_workers=0
    ):
        self.instance_list = instance_list
        self.graph_dir = graph_dir
        self.embedding_dir = embedding_dir if os.path.isdir(embedding_dir) else None
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
        if os.path.isfile(self.instance_list):
            self.load_list_file()
        if os.path.isdir(self.instance_list):
            self.load_list_dir()

    def load_list_file(self):
        for row in (open(self.instance_list)):
            entry_id = row.strip()
            if entry_id in self.ready_entries:
                continue
            print(f"Processing entry: {entry_id}")
            for (ch, data) in self.get_graph_from_entry_id(entry_id):
                if data:
                    torch.save(data, os.path.join(self.graph_dir, f"{entry_id}.{ch}.pt"))

    def load_list_dir(self):
        for file in os.listdir(self.instance_list):
            entry_id = file.split(".")[0]
            if entry_id in self.ready_entries:
                continue
            print(f"Processing file: {file}")
            for (ch, data) in self.get_graph_from_pdb_file(f"{self.instance_list}/{file}"):
                if data:
                    if file.endswith(".pdb") or file.endswith(".ent"):
                        file = ".".join(file.split(".")[0:-1])
                    tensor_file = os.path.join(self.graph_dir, f"{file}.{ch if ch!=' ' else '0'}.pt")
                    if os.path.isfile(tensor_file):
                        raise Exception(f"File {tensor_file} exists")
                    torch.save(data, tensor_file)

    def load_instances(self):
        embedding_list = set(
            [".".join(r.split(".")[0:-1]) for r in os.listdir(self.embedding_dir)] if self.embedding_dir else []
        )
        graph_files = [f"{self.graph_dir}/{r}" for r in os.listdir(self.graph_dir)]
        graph_files = [".".join(r.split("/")[-1].split(".")[0:-1]) for r in graph_files]
        for file in graph_files:
            if f"{file}" not in embedding_list:
                self.instances.append(f"{file}")
            else:
                print(f"Embedding {file} is ready")

    def get_graph_from_entry_id(self, pdb):
        cas, seqs = get_coords_for_pdb_id(pdb)
        return self.get_graphs(cas, seqs)

    def get_graph_from_pdb_file(self, pdb_file):
        cas, seqs = get_coords_for_pdb_file(pdb_file)
        return self.get_graphs(cas, seqs)

    def get_graphs(self, cas, seqs):
        graphs = []
        for ch in cas.keys():
            graphs.append((ch, self.get_chain_graph(cas[ch], seqs[ch])))
        return graphs

    def get_chain_graph(self, ca: list, sequence: str):
        structure = torch.from_numpy(np.asarray(ca))
        edge_index = gnn.radius_graph(
            structure, r=self.eps, loop=False, num_workers=self.num_workers
        )
        edge_index += 1  # shift for cls_idx
        x = torch.cat(
            [
                torch.LongTensor([self.esm_alphabet.cls_idx]),
                torch.LongTensor([
                    self.esm_alphabet.get_idx(res) for res in
                    self.esm_alphabet.tokenize(sequence)
                ]),
                torch.LongTensor([self.esm_alphabet.eos_idx]),
            ]
        )
        idx_mask = torch.zeros_like(x, dtype=torch.bool)
        idx_mask[1:-1] = True
        return Data(x=x, edge_index=edge_index, idx_mask=idx_mask)

    def len(self):
        return len(self.instances)

    def get(self, idx):
        data = torch.load(os.path.join(self.graph_dir, f"{self.instances[idx]}.pt"))
        return data, self.instances[idx]
