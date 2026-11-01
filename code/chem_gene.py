import pandas as pd

def get_chem_gene():
    path = "CTD_chem_gene_ixns.tsv.gz"

    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        comment="#",
        header=None,
        low_memory=False
    )

    df.columns = [
        "ChemicalID",
        "ChemicalName",
        "CasRN",
        "GeneSymbol",
        "GeneID",
        "GeneForm",
        "Organism",
        "OrganismID",
        "Interaction",
        "InteractionActions",
        "PubMedIDs"
    ]


    df.to_csv("CTD_chem_gene_ixns.tsv.csv", index=False)
  
    df = df[df["Organism"] == "Homo sapiens"]

    df = df[df["GeneForm"] == "gene"]

    df = df[["ChemicalID", "GeneID"]].dropna()

    df = df.drop_duplicates()


    return df

if __name__ == "__main__":
    df = get_chem_gene()
    print("Dataset created successfully!")
    print("Number of rows:", len(df))
    print(df.head())
