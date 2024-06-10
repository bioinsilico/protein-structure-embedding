import os
import itertools

import numpy as np
import torch
import esm
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Dataset

from scripts.utils.biopython_getter import get_coords_for_pdb_file
from scripts.utils.coords_getter import get_coords_for_pdb_id


def file_name(file):
    return os.path.splitext(file)[0]


class RcsbDataset(Dataset):
    def __init__(
            self,
            instance_list,
            graph_dir,
            embedding_dir=None,
            eps=8.0,
            esm_alphabet=esm.data.Alphabet.from_architecture("ESM-1b"),
            num_workers=0,
            granularity="chain"
    ):
        self.instance_list = instance_list
        self.graph_dir = graph_dir
        self.embedding_dir = embedding_dir if embedding_dir and os.path.isdir(embedding_dir) else None
        self.eps = eps
        self.esm_alphabet = esm_alphabet
        self.num_workers = num_workers
        self.granularity = "chain" if granularity != "entry" else "entry"
        self.entries = set({})
        self.instances = []
        self.ready_entries = set({})
        self.ready_list()
        self.load_list()
        self.load_instances()
        super().__init__()

    def ready_list(self):
        for file in os.listdir(self.graph_dir):
            self.ready_entries.add(file_name(file) if self.granularity == "entry" else file_name(file_name(file)))

    def load_list(self):
        if os.path.isfile(self.instance_list):
            self.load_list_file()
        elif os.path.isdir(self.instance_list):
            self.load_list_dir()
        else:
            raise Exception("--instance_list must be a valid file or directory")

    def load_list_file(self):
        for row in (open(self.instance_list)):
            entry_id = row.strip()
            self.entries.add(entry_id)
            if entry_id in self.ready_entries:
                continue
            print(f"Processing entry: {entry_id}")
            for (ch, data) in self.get_graph_from_entry_id(entry_id):
                if ch and data:
                    torch.save(data, os.path.join(self.graph_dir, f"{entry_id}.{ch}.pt"))
                elif data:
                    torch.save(data, os.path.join(self.graph_dir, f"{entry_id}.pt"))
                else:
                    raise Exception(f"Graph data is null for {entry_id}")

    def load_list_dir(self):
        for file in os.listdir(self.instance_list):
            entry_id = file_name(file)
            self.entries.add(entry_id)
            if entry_id in self.ready_entries:
                continue
            print(f"Processing file: {file}")
            for (ch, data) in self.get_graph_from_pdb_file(f"{self.instance_list}/{file}"):
                if file.endswith(".pdb") or file.endswith(".ent"):
                    file = file_name(file)
                if ch:
                    tensor_file = os.path.join(self.graph_dir, f"{file}.{ch if ch != ' ' else '0'}.pt")
                else:
                    tensor_file = os.path.join(self.graph_dir, f"{entry_id}.pt")
                if os.path.isfile(tensor_file):
                    raise Exception(f"File {tensor_file} exists")
                if data:
                    torch.save(data, tensor_file)
                else:
                    raise Exception(f"Graph data is null for {entry_id}")

    def load_instances(self):
        embedding_list = set(
            [file_name(file) for file in os.listdir(self.embedding_dir)] if self.embedding_dir else []
        )
        graph_files = [
            file_name(file) for file in os.listdir(self.graph_dir) if self.check_file(file)
        ]
        for file in graph_files:
            if f"{file}" not in embedding_list:
                self.instances.append(f"{file}")
            else:
                print(f"Embedding {file} is ready")

    def check_file(self, file):
        if self.granularity == "chain" and file_name(file_name(file)) in self.entries:
            return True
        if self.granularity == "entry" and file_name(file) in self.entries:
            return True
        return False

    def get_graph_from_entry_id(self, pdb):
        cas, seqs = get_coords_for_pdb_id(pdb)
        return self.get_graph_from_cas_adn_seqs(cas, seqs)

    def get_graph_from_pdb_file(self, pdb_file):
        cas, seqs = get_coords_for_pdb_file(pdb_file)
        return self.get_graph_from_cas_adn_seqs(cas, seqs)

    def get_graph_from_cas_adn_seqs(self, cas, seqs):
        if self.granularity == "chain":
            return self.get_multiple_graphs(cas, seqs)
        elif self.granularity == "entry":
            return self.get_single_graph(cas, seqs)

    def get_single_graph(self, cas, seqs):
        return [(None, self.get_chain_graph(
            list(itertools.chain.from_iterable(cas.values())),
            "".join(seqs.values())
        ))]

    def get_multiple_graphs(self, cas, seqs):
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

if __name__ == "__main__":
    dataset = RcsbDataset(
        "/Users/joan/data/structure-embedding/rcsb/csm-list-test.tsv",
        "/Users/joan/data/structure-embedding/rcsb/graph-csm"
    )
    print(len(dataset))