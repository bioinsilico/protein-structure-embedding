import os
import math
import warnings

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from Bio.PDB import PDBParser, MMCIFParser

from scripts.rcsb_datasets.rcsb_dataset import file_name
import requests
import gzip
import io


def get_coordinates(res):
    return res["CA"].get_coord()


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
    return math.exp(-(d - 3.6) ** 2 / 12)


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


def get_res_attr(residues):
    angles = [get_angles(res.internal_coord) for res in residues]
    distances = [get_distance(idx, residues) for idx, res in enumerate(residues)]
    contacts = get_contacts(residues)
    coords = [get_coordinates(res) for res in residues]
    if len(angles) != len(distances) or len(angles) != len(coords):
        raise Exception("CA number missmatch")
    graph_nodes = torch.tensor([[
        x, y, z,
        math.sin(a),
        math.cos(a),
        math.sin(b),
        math.cos(b),
        math.sin(c),
        math.cos(c),
        f, b
    ] for (x, y, z), (a, b, c), (f, b) in list(zip(coords, angles, distances))], dtype=torch.float)
    graph_edges = torch.tensor([
        c for c, d in contacts
    ], dtype=torch.int64)
    edge_attr = torch.tensor([
        d for c, d in contacts
    ])
    return graph_nodes, graph_edges, edge_attr


def get_geo_from_entry_structure(structure):
    geos = []
    residues = []
    for ch in structure:
        ch.atom_to_internal_coordinates(verbose=True)
        ch_residues = [res for res in ch.get_residues() if "CA" in res]
        if len(ch_residues) < 6:
            continue
        residues.extend(ch_residues)
    graph_nodes, graph_edges, edge_attr = get_res_attr(residues)
    geos.append((None, Data(
        graph_nodes,
        edge_index=graph_edges.t().contiguous(),
        edge_attr=edge_attr
    )))
    return geos


def get_geo_from_single_chain_structure(structure):
    geos = []
    for ch in structure:
        ch.atom_to_internal_coordinates(verbose=True)
        residues = [res for res in ch.get_residues() if "CA" in res]
        if len(residues) < 6:
            continue
        graph_nodes, graph_edges, edge_attr = get_res_attr(residues)
        geos.append((ch.id, Data(
            graph_nodes,
            edge_index=graph_edges.t().contiguous(),
            edge_attr=edge_attr
        )))
    return geos


class RcsbGeoDataset(Dataset):
    def __init__(
            self,
            instance_list,
            geo_dir,
            eps=8.0,
            num_workers=0,
            granularity="chain"
    ):
        self.instance_list = instance_list
        self.geo_dir = geo_dir
        self.eps = eps
        self.num_workers = num_workers
        self.granularity = "chain" if granularity != "entry" else "entry"
        self.instances = []
        self.ready_entries = set({})
        self.ready_list()
        self.load_list()
        super().__init__()

    def ready_list(self):
        for file in os.listdir(self.geo_dir):
            self.ready_entries.add(file_name(file) if self.granularity == "entry" else file_name(file_name(file)))
        print(
            f"Found {len(self.ready_entries)} ready entries in {self.geo_dir}" +
            (f": {next(iter(self.ready_entries))}, ..." if len(self.ready_entries) > 0 else "")
        )

    def load_list(self):
        if os.path.isfile(self.instance_list):
            self.load_list_file()
        if os.path.isdir(self.instance_list):
            self.load_list_dir()

    def load_list_file(self):
        for pdb in open(self.instance_list):
            pdb = pdb.strip()
            if pdb in self.ready_entries:
                continue
            print(f"Processing PDB: {pdb}")
            try:
                for (ch, data) in self.get_geo_graph_from_pdb_entry(pdb.lower()):
                    if data is None:
                        continue
                    file = pdb
                    if ch is not None:
                        file = f"{file}.{ch}"
                    torch.save(
                        data,
                        os.path.join(self.geo_dir, f"{file}.pt")
                    )
            except Exception:
                warnings.warn(f"PDB {pdb} failed")

    def load_list_dir(self):
        for file in os.listdir(self.instance_list):
            if file in self.ready_entries:
                continue
            print(f"Processing file: {file}")
            for (ch, data) in self.get_geo_graph_from_pdb_file(f"{self.instance_list}/{file}"):
                if data is None:
                    continue
                if file.endswith(".pdb") or file.endswith(".ent"):
                    file = file_name(file)
                if ch is not None:
                    file = f"{file}.{ch}"
                torch.save(
                    data,
                    os.path.join(self.geo_dir, f"{file}.pt")
                )

    def get_geo_graph_from_pdb_entry(self, pdb):
        pdb_url = f"https://files.rcsb.org/download/{pdb}.cif.gz"
        response = requests.get(pdb_url)
        response.raise_for_status()
        parser = MMCIFParser()
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
            return self.get_geo_from_structure(parser.get_structure(pdb, io.TextIOWrapper(gz_file))[0])

    def get_geo_graph_from_pdb_file(self, pdb_file):
        parser = PDBParser()
        structure = parser.get_structure("structure", pdb_file)
        return self.get_geo_from_structure(structure[0])

    def get_geo_from_structure(self, structure):
        if self.granularity == "chain":
            return get_geo_from_single_chain_structure(structure)
        elif self.granularity == "entry":
            return get_geo_from_entry_structure(structure)

    def get_instance(self, idx):
        return self.instances[idx]

    def len(self):
        return len(self.instances)

    def get(self, idx):
        data = torch.load(os.path.join(self.geo_dir, f"{self.instances[idx]}.pt"))
        return data
