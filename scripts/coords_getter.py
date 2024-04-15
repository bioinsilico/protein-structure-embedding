from Bio.Data.PDBData import protein_letters_3to1_extended
from rcsb.utils.io.MarshalUtil import MarshalUtil


aas = ["GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TYR", "TRP", "PRO", "HIS",
       "LYS", "ARG", "SER", "THR", "GLU", "GLN", "ASP", "ASN", "CYS", "MET"]
min_num_residues = 10


def get_poly_entities(data_container):
    entity = data_container.getObj('entity')
    poly_ent_ids = []
    for i in range(0, len(entity.data)):
        d_row = entity.getRowAttributeDict(i)
        ent_id = d_row["id"]
        ent_type = d_row["type"]
        if ent_type == "polymer":
            poly_ent_ids.append(ent_id)
    return poly_ent_ids


def get_ca_coords(data_container, poly_ent_ids: list):
    atom_sites = data_container.getObj('atom_site')
    coords_current_chain = []
    polychains_coords = {}
    polychains_seqs = {}
    current_seq = []
    asym_id = None
    for i in range(0, len(atom_sites.data)):
        d_row = atom_sites.getRowAttributeDict(i)
        ent_id = d_row["label_entity_id"]
        if ent_id not in poly_ent_ids:
            continue
        current_asym_id = d_row["label_asym_id"]
        if asym_id is not None and current_asym_id != asym_id:
            # we'll keep only chains with at least this residues. This should also filter out dna/rna
            if len(coords_current_chain) >= min_num_residues:
                polychains_coords[asym_id] = coords_current_chain
                polychains_seqs[asym_id] = "".join(current_seq)
            coords_current_chain = []
            current_seq = []
        aa_type = d_row["label_comp_id"]
        atom_type = d_row["label_atom_id"]
        model_num = d_row["pdbx_PDB_model_num"]
        alt_loc = d_row["label_alt_id"]
        if atom_type == "CA" and model_num == 1 and aa_type in aas and (alt_loc == "." or alt_loc == "A"):
            coords_current_chain.append([d_row["Cartn_x"], d_row["Cartn_y"], d_row["Cartn_z"]])
            current_seq.append(protein_letters_3to1_extended[aa_type])
        asym_id = current_asym_id
    if len(coords_current_chain) >= min_num_residues and asym_id not in polychains_coords:
        polychains_coords[asym_id] = coords_current_chain
        polychains_seqs[asym_id] = "".join(current_seq)
    return polychains_coords, polychains_seqs


def get_coords_from_file(cif_file_url):
    """
    Get 2 dictionaries: 1) coordinates, with keys asym_ids of protein polymer entities and values the
    coordinates as a list of 3-size lists; 2) sequences, with keys asym_ids of protein polymer entities and values the
    sequences (modelled residues and only standard aas)
    If no protein polymer entities found the dictionary will be empty.
    :param cif_file_url: a link to a bcif file
    :return:
    """

    mU = MarshalUtil()
    data_container_list = mU.doImport(cif_file_url, fmt="bcif")
    data_container = data_container_list[0]
    poly_ent_ids = get_poly_entities(data_container)
    chain_coords, chain_seqs = get_ca_coords(data_container, poly_ent_ids)

    return chain_coords, chain_seqs


def get_coords_for_pdb_id(pdb_id):
    url = "https://models.rcsb.org/%s.bcif" % pdb_id
    chain_coords, chain_seqs = get_coords_from_file(url)
    print("Found %d valid protein chains in %s. Asym_ids are : %s" % (len(chain_coords), pdb_id, ",".join(chain_coords.keys())))
    return chain_coords, chain_seqs


def main():
    chain_coords, chain_seqs = get_coords_for_pdb_id("2trx")
    print(chain_coords.keys())


if __name__ == "__main__":
    main()