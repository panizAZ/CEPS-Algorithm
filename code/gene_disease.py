import pandas as pd
import os

def get_gene_disease():
    path = "CTD_genes_diseases.tsv.gz"

    chunksize = 200_000
    edges = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        comment="#",
        header=None,
        chunksize=chunksize,
        low_memory=False
    ):
        chunk.columns = [
            "GeneSymbol",
            "GeneID",
            "DiseaseName",
            "DiseaseID",
            "DirectEvidence",
            "InferenceGeneSymbol",
            "InferenceScore",
            "OmimIDs",
            "PubMedIDs"
        ]

        chunk = chunk[chunk["GeneID"].notna()]
        chunk = chunk[chunk["DirectEvidence"].notna()]


        chunk = chunk[["GeneID", "GeneSymbol", "DiseaseID", "DiseaseName"]]

        edges.append(chunk)

    df = pd.concat(edges).drop_duplicates()
    return df

if __name__ == "__main__":
    df = get_gene_disease()
    print("Gene-Disease dataset created successfully!")
    print("Number of edges:", len(df))
    print(df.head())

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "gene_disease_edges.csv")
    df.to_csv(output_path, index=False)
    print("File saved to:", output_path)