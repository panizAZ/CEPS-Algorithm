import CEPS

def initialize_if_needed():
    if CEPS.z is None:
        CEPS.initialize()

def get_chemical_index(chemical_name):

    initialize_if_needed()

    for local_idx, name in CEPS.data.idx2chemical.items():
        if name == chemical_name:
            return local_idx + CEPS.offsets["chemical"]

    raise ValueError(f"Chemical '{chemical_name}' not found")


def get_disease_index(disease_name):

    initialize_if_needed()

    for local_idx, name in CEPS.data.idx2disease.items():
        if name == disease_name:
            return local_idx + CEPS.offsets["disease"]

    raise ValueError(f"Disease '{disease_name}' not found")