import torch
from torch_geometric.data import HeteroData
from merge_data import merge

def graph_hetero():
    ceps_df = merge()

    if ceps_df.empty:
        print("WARNING: Input dataset is empty. Graph not created.")
        return None

    print("Input dataset shape:", ceps_df.shape)

    ceps_df["ChemicalID"] = ceps_df["ChemicalID"].astype(str)
    ceps_df["GeneID"] = ceps_df["GeneID"].astype(str)
    ceps_df["MONDO"] = ceps_df["MONDO"].astype(int)

    chemical_ids = ceps_df["ChemicalID"].unique()
    gene_ids = ceps_df["GeneID"].unique()
    disease_ids = ceps_df["MONDO"].unique()

    chemical_id_to_name = {c: c for c in chemical_ids}


    if 'GeneSymbol' in ceps_df.columns:
        gene_id_to_name = ceps_df.groupby('GeneID')['GeneSymbol'].first().to_dict()
    else:
        gene_id_to_name = {g: g for g in gene_ids}

    if 'DiseaseName' in ceps_df.columns:
        disease_id_to_name = ceps_df.groupby('MONDO')['DiseaseName'].first().to_dict()
    else:
        disease_id_to_name = {d: str(d) for d in disease_ids}

    chemical2idx = {c: i for i, c in enumerate(chemical_ids)}
    gene2idx = {g: i for i, g in enumerate(gene_ids)}
    disease2idx = {d: i for i, d in enumerate(disease_ids)}

    chem_gene_src = torch.tensor(
        ceps_df["ChemicalID"].map(chemical2idx).values, dtype=torch.long
    )
    chem_gene_dst = torch.tensor(
        ceps_df["GeneID"].map(gene2idx).values, dtype=torch.long
    )

    gene_disease_src = torch.tensor(
        ceps_df["GeneID"].map(gene2idx).values, dtype=torch.long
    )
    gene_disease_dst = torch.tensor(
        ceps_df["MONDO"].map(disease2idx).values, dtype=torch.long
    )

    data = HeteroData()
    data["chemical"].num_nodes = len(chemical_ids)
    data["gene"].num_nodes = len(gene_ids)
    data["disease"].num_nodes = len(disease_ids)

    data[("chemical", "chemical_gene", "gene")].edge_index = torch.stack(
        [chem_gene_src, chem_gene_dst]
    )
    data[("gene", "gene_disease", "disease")].edge_index = torch.stack(
        [gene_disease_src, gene_disease_dst]
    )


    data.idx2chemical = {i: chemical_id_to_name[chem_id] for i, chem_id in enumerate(chemical_ids)}
    data.idx2gene = {i: gene_id_to_name[gene_id] for i, gene_id in enumerate(gene_ids)}
    data.idx2disease = {i: disease_id_to_name[disease_id] for i, disease_id in enumerate(disease_ids)}


    data_homo = data.to_homogeneous()

    hom_idx2name = {}
    for node_type_str, start_idx in [("chemical", 0),
                                     ("gene", data["chemical"].num_nodes),
                                     ("disease", data["chemical"].num_nodes + data["gene"].num_nodes)]:
        n_nodes = data[node_type_str].num_nodes
        for i in range(n_nodes):
            hom_idx = start_idx + i
            if node_type_str == "chemical":
                hom_idx2name[hom_idx] = f"{data.idx2chemical[i]} (Chemical)"
            elif node_type_str == "gene":
                hom_idx2name[hom_idx] = f"{data.idx2gene[i]} (Gene)"
            elif node_type_str == "disease":
                hom_idx2name[hom_idx] = f"{data.idx2disease[i]} (Disease)"

    data_homo.hom_idx2name = hom_idx2name
    data_homo.offsets = {
        "chemical": 0,
        "gene": data["chemical"].num_nodes,
        "disease": data["chemical"].num_nodes + data["gene"].num_nodes
    }

    if data_homo.num_nodes > 0 and data_homo.num_edges > 0:
        print("Graph successfully created.")
        print("Total nodes:", data_homo.num_nodes)
        print("Total edges:", data_homo.num_edges)
        print("Edge types:", data_homo.edge_type.unique())
    else:
        print("Graph creation failed.")

    return data_homo

if __name__ == "__main__":
    data_homo = graph_hetero()
    if data_homo is not None:
        torch.save(data_homo, "hetero_graph.pt")
        print("Graph + full metadata saved successfully")