import random

from Bio.PDB import PDBParser
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.Polypeptide import is_aa
from collections import OrderedDict


def identity(*lists):
    return lists


def get_coords_for_pdb_file(pdb_file, add_noise=identity):
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    return __get_graph_from_structure(structure[0], add_noise)


def __get_graph_from_structure(structure, add_noise=identity):
    chains = [s.id for s in structure.get_chains()]
    ca_map = OrderedDict()
    seq_map = OrderedDict()
    label_sed_ids = OrderedDict()
    for ch in chains:
        ca_atoms = [atom for atom in structure.get_atoms() if
                    atom.get_name() == "CA" and is_aa(atom.parent.resname) and atom.parent.parent.id == ch]
        if len(ca_atoms) < 10:
            continue
        __ca_map, __seq_map, __label_sed_ids = add_noise(
            [atom.get_coord() for atom in ca_atoms],
            [protein_letters_3to1_extended[c.parent.resname] for c in ca_atoms],
            [atom.full_id[3][1] for atom in ca_atoms]
        )
        ca_map[ch] = __ca_map
        seq_map[ch] = "".join(__seq_map)
        label_sed_ids[ch] = __label_sed_ids
    return ca_map, seq_map, label_sed_ids


def remove_random_residues(n):
    def __remove_n_random_elements_from_multiple_lists(*lists):
        if not lists:
            raise ValueError("At least one list must be provided")
        list_lengths = [len(lst) for lst in lists]
        if any(length < n for length in list_lengths):
            raise ValueError("n cannot be greater than the length of any of the lists")
        indices_to_remove = random.sample(range(list_lengths[0]), n)
        print(indices_to_remove)
        for lst in lists:
            for index in sorted(indices_to_remove, reverse=True):
                del lst[index]
        return lists

    return __remove_n_random_elements_from_multiple_lists
