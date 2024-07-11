import os
import math

import torch
from torch_geometric.data import Data, Dataset

from scripts.rcsb_datasets.geometry import angle_between_points, distance_between_points, exp_distance, \
    angle_between_planes, angle_between_four_points
from scripts.rcsb_datasets.rcsb_dataset import file_name
from scripts.utils.biopython_getter import get_coords_for_pdb_file

from scripts.utils.coords_getter import get_coords_for_pdb_id


def get_angles(idx, residues):
    if 0 < idx < len(residues) - 1:
        res_0, idx_0 = residues[idx]
        res_f, idx_f = residues[idx + 1]
        res_b, idx_b = residues[idx - 1]
        if idx_0 - idx_b == 1 and idx_f - idx_0 == 1:
            a = angle_between_points(res_b, res_0, res_f)
            return (
                math.sin(a),
                math.cos(a)
            )
    return 0, 0


def contiguous(idx_i, idx_j):
    if abs(idx_i - idx_j) == 1:
        return 1
    return 0


def edge_angles(idx_i, idx_j, residues):
    if (
            (0 < idx_i < len(residues) - 1 and 0 < idx_j < len(residues) - 1) and
            (residues[idx_i][1] - residues[idx_i-1][1] == 1) and
            (residues[idx_i+1][1] - residues[idx_i][1] == 1) and
            (residues[idx_j][1] - residues[idx_j-1][1] == 1) and
            (residues[idx_j+1][1] - residues[idx_j][1] == 1)
    ):
        a = angle_between_planes(
            (residues[idx_i-1][0], residues[idx_i][0], residues[idx_i+1][0]),
            (residues[idx_j-1][0], residues[idx_j][0], residues[idx_j+1][0])
        )
        return (
            math.sin(a),
            math.cos(a)
        )
    return 0, 0


def orientation_angles(idx_i, idx_j, residues):
    fs = 0
    fc = 0
    bs = 0
    bc = 0
    if (
            (idx_i < len(residues) - 1) and
            (idx_j < len(residues) - 1) and
            (residues[idx_i + 1][1] - residues[idx_i][1] == 1) and
            (residues[idx_j + 1][1] - residues[idx_j][1] == 1)
    ):
        a = angle_between_four_points(
            residues[idx_i][0], residues[idx_i + 1][0],
            residues[idx_j][0], residues[idx_j + 1][0]
        )
        fs = math.sin(a)
        fc = math.cos(a)
    if (
            (idx_i > 0 and idx_j > 0) and
            (residues[idx_i][1] - residues[idx_i-1][1] == 1) and
            (residues[idx_j][1] - residues[idx_j-1][1] == 1)
    ):
        a = angle_between_four_points(
            residues[idx_i - 1][0], residues[idx_i][0],
            residues[idx_j - 1][0], residues[idx_j][0]
        )
        bs = math.sin(a)
        bc = math.cos(a)
    return fs, fc, bs, bc


def get_contacts(residues):
    contact_map = []
    for i, (res_i, idx_i) in enumerate(residues):
        for j, (res_j, idx_j) in enumerate(residues):
            if i == j:
                continue
            d = distance_between_points(res_i, res_j)
            if d < 8.:
                e = exp_distance(d)
                c = contiguous(idx_i, idx_j)
                es, ec = edge_angles(i, j, residues)
                fs, fc, bs, bc = orientation_angles(i, j, residues)
                contact_map.append(([i, j], (e, c, es, ec, fs, fc, bs, bc)))
    return contact_map


def get_res_attr(residues):
    angles = [get_angles(idx, residues) for idx, res in enumerate(residues)]
    contacts = get_contacts(residues)
    graph_nodes = torch.tensor([[
        a, b
    ] for a, b in angles], dtype=torch.float)
    graph_edges = torch.tensor([
        c for c, d in contacts
    ], dtype=torch.int64)
    edge_attr = torch.tensor([
        d for c, d in contacts
    ])
    return graph_nodes, graph_edges, edge_attr


def get_geo_from_entry_structure(chain_coords):
    geos = []
    residues = []
    for ch, ch_residues in chain_coords.items():
        residues.extend(ch_residues)
    graph_nodes, graph_edges, edge_attr = get_res_attr(residues)
    geos.append((None, Data(
        graph_nodes,
        edge_index=graph_edges.t().contiguous(),
        edge_attr=edge_attr
    )))
    return geos


def get_geo_from_single_chain_structure(chain_coords):
    geos = []
    for ch, residues in chain_coords.items():
        graph_nodes, graph_edges, edge_attr = get_res_attr(residues)
        geos.append((ch, Data(
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
        chain_coords, chain_seqs, chain_label_seqs = get_coords_for_pdb_id(pdb)
        return self.get_geo_graph(chain_coords, chain_label_seqs)

    def get_geo_graph_from_pdb_file(self, pdb_file):
        chain_coords, chain_seqs, chain_label_seqs = get_coords_for_pdb_file(pdb_file)
        return self.get_geo_graph(chain_coords, chain_label_seqs)

    def get_geo_graph(self, chain_coords, chain_label_seqs):
        chain_label_coords = dict([(ch, list(zip(chain_coords[ch], chain_label_seqs[ch]))) for ch in chain_coords])
        if self.granularity == "chain":
            return get_geo_from_single_chain_structure(chain_label_coords)
        elif self.granularity == "entry":
            return get_geo_from_entry_structure(chain_label_coords)

    def get_instance(self, idx):
        return self.instances[idx]

    def len(self):
        return len(self.instances)

    def get(self, idx):
        data = torch.load(os.path.join(self.geo_dir, f"{self.instances[idx]}.pt"))
        return data
