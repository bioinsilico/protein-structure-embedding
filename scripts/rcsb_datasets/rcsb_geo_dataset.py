import os
import math

import numpy as np
import torch
import esm
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Dataset
from Bio.PDB import PDBParser
from scripts.utils.coords_getter import get_coords_for_pdb_id


def get_angles(res_internal_coord):
    psi = res_internal_coord.get_angle("psi")
    phi = res_internal_coord.get_angle("phi")
    omega = res_internal_coord.get_angle("omega")
    return (
        psi / 180 * math.pi if psi else 0.,
        phi / 180 * math.pi if phi else 0.,
        omega / 180 * math.pi if omega else 0.
    )


def c_alpha_dist(ri, rj):
    if "CA" in ri and "CA" in rj:
        one = ri["CA"].get_coord()
        two = rj["CA"].get_coord()
        return np.linalg.norm(one - two)
    return 3.8


def get_distance(idx, residues):
    forward = 0.
    backward = 0.
    if idx < len(residues) - 1:
        forward = c_alpha_dist(residues[idx], residues[idx + 1])
    if idx > 0:
        backward = c_alpha_dist(residues[idx - 1], residues[idx])
    return forward, backward


class RcsbGeoDataset(Dataset):
    def __init__(
            self,
            instance_list,
            geo_dir,
            eps=8.0,
            esm_alphabet=esm.data.Alphabet.from_architecture("ESM-1b"),
            num_workers=0
    ):
        self.instance_list = instance_list
        self.geo_dir = geo_dir
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
        for row in os.listdir(self.geo_dir):
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
            try:
                for (ch, data) in self.get_graph_from_entry_id(entry_id):
                    if data:
                        torch.save(data, os.path.join(self.geo_dir, f"{entry_id}.{ch}.pt"))
            except:
                print(f"Entry {entry_id} failed")

    def load_list_dir(self):
        for file in os.listdir(self.instance_list):
            print(f"Processing file: {file}")
            for (ch, data) in self.get_geo_from_pdb_file(f"{self.instance_list}/{file}"):
                if data:
                    if file.endswith(".pdb"):
                        file = file.split(".")[0]
                    torch.save(
                        torch.from_numpy(np.array(data)),
                        os.path.join(self.geo_dir, f"{file}.pt")
                    )

    def load_instances(self):
        embedding_list = set(
            [".".join(r.split(".")[0:-1]) for r in os.listdir(self.geo_dir)]
        )
        graph_files = [f"{self.geo_dir}/{r}" for r in os.listdir(self.geo_dir)]
        graph_files = [".".join(r.split("/")[-1].split(".")[0:-1]) for r in sorted(graph_files, key=os.path.getsize)]
        for file in graph_files:
            if f"{file}" not in embedding_list:
                self.instances.append(f"{file}")
            else:
                print(f"Embedding {file} is ready")

    def get_graph_from_entry_id(self, pdb):
        cas, seqs = get_coords_for_pdb_id(pdb)
        graphs = []
        for ch in cas.keys():
            graphs.append((ch, self.get_chain_graph(cas[ch], seqs[ch])))
        return graphs

    def get_geo_from_pdb_file(self, pdb_file):
        parser = PDBParser()
        structure = parser.get_structure("structure", pdb_file)
        return self.get_geo_from_structure(structure[0])

    def get_geo_from_structure(self, structure):
        geos = []
        for ch in structure:
            ch.atom_to_internal_coordinates(verbose=True)
            angles = [get_angles(res.internal_coord) for res in ch.get_residues()]
            residues = list(ch.get_residues())
            distances = [get_distance(idx, residues) for idx, res in enumerate(residues)]
            if len(angles) != len(distances):
                raise Exception("CA number missmatch")
            geos.append((ch.id, [(a, b, c, x, y) * 16 for (a, b, c), (x, y) in list(zip(angles, distances))]))
        return geos

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

    def get_instance(self, idx):
        return self.instances[idx]

    def len(self):
        return len(self.instances)

    def get(self, idx):
        data = torch.load(os.path.join(self.geo_dir, f"{self.instances[idx]}.pt"))
        return data
