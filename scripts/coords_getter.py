import os
import urllib.request
from mmcif.io.IoAdapterCore import IoAdapterCore


aas = ["GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TYR", "TRP", "PRO", "HIS",
       "LYS", "ARG", "SER", "THR", "GLU", "GLN", "ASP", "ASN", "CYS", "MET"]


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


def get_entity_to_asym_map(data_container):
    atom_sites = data_container.getObj('atom_site')
    entity_to_asym = {}
    for i in range(0, len(atom_sites.data)):
        d_row = atom_sites.getRowAttributeDict(i)
        ent_id = d_row["label_entity_id"]
        current_asym_id = d_row["label_asym_id"]
        if ent_id not in entity_to_asym:
            entity_to_asym[ent_id] = {current_asym_id}
        else:
            entity_to_asym[ent_id].add(current_asym_id)
    return entity_to_asym


def get_ca_coords(data_container, asym_id: str):

    atom_sites = data_container.getObj('atom_site')

    coords = []
    for i in range(0, len(atom_sites.data)):
        d_row = atom_sites.getRowAttributeDict(i)
        current_asym_id = d_row["label_asym_id"]
        if current_asym_id != asym_id:
            continue
        aa_type = d_row["label_comp_id"]
        atom_type = d_row["label_atom_id"]
        model_num = int(d_row["pdbx_PDB_model_num"])
        if aa_type not in aas:
            continue
        if atom_type != "CA":
            continue
        if model_num != 1:
            continue
        coords.append([d_row["Cartn_x"], d_row["Cartn_y"], d_row["Cartn_z"]])
    return coords


def get_coords_from_file(local_cif_file):
    """
    Get a dictionary with keys asym_ids of protein polymer entities and values the coordinates as a list of 3-size lists.
    If no protein polymer entities found the dictionary will be empty.
    :param local_cif_file:
    :return:
    """
    io = IoAdapterCore()
    list_data_container = io.readFile(local_cif_file)
    data_container = list_data_container[0]

    poly_ent_ids = get_poly_entities(data_container)
    entity_to_asym = get_entity_to_asym_map(data_container)

    chain_coords = {}
    for ent_id in poly_ent_ids:
        for asym_id in entity_to_asym[ent_id]:
            cas = get_ca_coords(data_container, asym_id)
            if len(cas) != 0:
                chain_coords[asym_id] = cas
    return chain_coords


def get_coords_for_pdb_id(pdb_id, tmp_dir):
    url = "https://files.rcsb.org/download/%s.cif.gz" % pdb_id
    filepath_local = os.path.join(tmp_dir, "%s.cif.gz" % pdb_id)
    print("Downloading %s to %s" % (url, filepath_local))
    urllib.request.urlretrieve(url, filepath_local)
    chain_coords = get_coords_from_file(filepath_local)
    os.remove(filepath_local)
    return chain_coords


def main():

    chain_coords = get_coords_for_pdb_id("3hbx", "/tmp")
    print(chain_coords.keys())


if __name__ == "__main__":
    main()