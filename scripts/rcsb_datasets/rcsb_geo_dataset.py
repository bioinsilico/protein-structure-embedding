import os
import math

import numpy as np
import torch
import esm
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Dataset
from Bio.PDB import PDBParser

from scripts.rcsb_datasets.rcsb_dataset import file_name
from scripts.utils.coords_getter import get_coords_for_pdb_id


def get_coordinates(res):
    res["CA"].get_coord()


def get_angles(res_internal_coord):
    psi = res_internal_coord.get_angle("psi")
    phi = res_internal_coord.get_angle("phi")
    omega = res_internal_coord.get_angle("omega")
    return (
        psi / 180 * math.pi if psi else 0.,
        phi / 180 * math.pi if phi else 0.,
        omega / 180 * math.pi if omega else 0.
    )


def distance_exp(d):
    return math.exp(-(d-3.6)**2 / 12)


def c_alpha_dist(ri, rj):
    d = 3.8
    if "CA" in ri and "CA" in rj:
        one = ri["CA"].get_coord()
        two = rj["CA"].get_coord()
        d = np.linalg.norm(one - two)
    return distance_exp(d)


def get_distance(idx, residues):
    forward = 0.
    backward = 0.
    if idx < len(residues) - 1:
        forward = c_alpha_dist(residues[idx], residues[idx + 1])
    if idx > 0:
        backward = c_alpha_dist(residues[idx - 1], residues[idx])
    return forward, backward


def get_distance_pair(ri, rj):
    if "CA" in ri and "CA" in rj:
        one = ri["CA"].get_coord()
        two = rj["CA"].get_coord()
        return np.linalg.norm(one - two)
    return 8.


def get_contacts(residues):
    map = []
    for i, res_i in enumerate(residues):
        for j, res_j in enumerate(residues):
            if i == j:
                continue
            d = get_distance_pair(res_i, res_j)
            if d < 8.:
                map.append(([i, j], distance_exp(d)))
    return map


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
            for (ch, data) in self.get_geo_graph_from_pdb_file(f"{self.instance_list}/{file}"):
                if data:
                    if file.endswith(".pdb") or file.endswith(".ent"):
                        file = file_name(file)
                    torch.save(
                        data,
                        os.path.join(self.geo_dir, f"{file}.{ch}.pt")
                    )

    def get_graph_from_entry_id(self, pdb):
        cas, seqs = get_coords_for_pdb_id(pdb)
        graphs = []
        for ch in cas.keys():
            graphs.append((ch, self.get_chain_graph(cas[ch], seqs[ch])))
        return graphs

    def get_geo_graph_from_pdb_file(self, pdb_file):
        parser = PDBParser()
        structure = parser.get_structure("structure", pdb_file)
        return self.get_geo_from_structure(structure[0])

    def get_geo_from_structure(self, structure):
        geos = []
        for ch in structure:
            ch.atom_to_internal_coordinates(verbose=True)
            residues = [res for res in ch.get_residues() if res.internal_coord is not None]
            angles = [get_angles(res.internal_coord) for res in residues]
            distances = [get_distance(idx, residues) for idx, res in enumerate(residues)]
            contacts = get_contacts(residues)
            if len(angles) != len(distances):
                raise Exception("CA number missmatch")
            graph_nodes = torch.tensor([[
                math.sin(a),
                math.cos(a),
                math.sin(b),
                math.cos(b),
                math.sin(c),
                math.cos(c),
                x,
                y
            ] for (a, b, c), (x, y) in list(zip(angles, distances))], dtype=torch.float)
            graph_edges = torch.tensor([
                c for c, d in contacts
            ], dtype=torch.int64)
            graph_edge_attr = torch.tensor([
                [d] for c, d in contacts
            ], dtype=torch.float)
            geos.append((ch.id, Data(
                graph_nodes,
                edge_index=graph_edges.t().contiguous(),
                edge_attr=graph_edge_attr
            )))
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
