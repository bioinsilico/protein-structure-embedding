from Bio.PDB import PDBParser
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.Polypeptide import is_aa
from collections import OrderedDict


def get_coords_for_pdb_file(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file)
    return __get_graph_from_structure(structure[0])


def __get_graph_from_structure(structure):
    chains = [s.id for s in structure.get_chains()]
    ca_map = OrderedDict()
    seq_map = OrderedDict()
    label_sed_ids = OrderedDict()
    for ch in chains:
        ca_atoms = [atom for atom in structure.get_atoms() if
                    atom.get_name() == "CA" and is_aa(atom.parent.resname) and atom.parent.parent.id == ch]
        if len(ca_atoms) == 0:
            continue
        ca_map[ch] = [atom.get_coord() for atom in ca_atoms]
        seq_map[ch] = "".join([protein_letters_3to1_extended[c.parent.resname] for c in ca_atoms])
        label_sed_ids[ch] = [atom.full_id[3][1] for atom in ca_atoms]
    return ca_map, seq_map, label_sed_ids
