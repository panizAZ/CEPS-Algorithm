from chem_gene import *
from gene_disease import *

import pandas as pd
from collections import defaultdict, deque

CANCER_ROOT = "0004992"


def merge():

    ceg = get_chem_gene()
    cgd = get_gene_disease()

    ceg["GeneID"] = ceg["GeneID"].astype(str)
    cgd["GeneID"] = cgd["GeneID"].astype(str)

    ceps_raw = ceg.merge(cgd, on="GeneID", how="inner")
    print("After Gene merge:", ceps_raw.shape)

    if ceps_raw.empty:
        print("WARNING: CEPS raw is empty after Gene merge")
        return ceps_raw

    mask = ceps_raw["DiseaseID"].astype(str).str.match(r"^(MESH:|OMIM:)")
    ceps_raw = ceps_raw[mask].copy()
    ceps_raw["DiseaseID_norm"] = ceps_raw["DiseaseID"]


    mondo_xref = pd.read_csv("mondo/mondo_references.csv", dtype=str)

    mondo_xref = mondo_xref[
        mondo_xref["ontology"].isin(["OMIM", "MESH"])
    ][["ontology", "ontology_id", "mondo_id"]]

    mondo_xref["DiseaseID_norm"] = (
        mondo_xref["ontology"] + ":" + mondo_xref["ontology_id"]
    )

    mondo_xref = mondo_xref[["DiseaseID_norm", "mondo_id"]]
    mondo_xref.columns = ["DiseaseID_norm", "MONDO"]


    ceps = ceps_raw.merge(
        mondo_xref,
        on="DiseaseID_norm",
        how="left"
    )

    print("Before MONDO drop:", ceps.shape)
    print("Mapped MONDO count:", ceps["MONDO"].notna().sum())

    unmapped = ceps[ceps["MONDO"].isna()][["DiseaseID"]].drop_duplicates()
    if not unmapped.empty:
        unmapped.to_csv("unmapped_diseases.csv", index=False)
        print(f"Unmapped diseases saved: {len(unmapped)}")

    ceps = ceps.dropna(subset=["MONDO"])
    if ceps.empty:
        print("WARNING: All rows dropped after MONDO mapping")
        return ceps

    ceps["MONDO"] = ceps["MONDO"].str.replace(r"^MONDO:", "", regex=True)


    parents = pd.read_csv("mondo/mondo_parents.csv", dtype=str)

    parents = parents[
        parents["parent"].str.fullmatch(r"\d+") &
        parents["child"].str.fullmatch(r"\d+")
    ]

    graph = defaultdict(list)
    for _, row in parents.iterrows():
        graph[row["parent"]].append(row["child"])

    print("Children of cancer root:", graph.get(CANCER_ROOT, [])[:10])

    def get_descendants(root):
        seen = set()
        q = deque([root])
        while q:
            node = q.popleft()
            for child in graph.get(node, []):
                if child not in seen:
                    seen.add(child)
                    q.append(child)
        return seen

    cancer_mondos = get_descendants(CANCER_ROOT)
    cancer_mondos.add(CANCER_ROOT)

    print("Cancer set size:", len(cancer_mondos))
    print("Unique MONDOs before filter:", ceps["MONDO"].nunique())

    ceps = ceps[ceps["MONDO"].isin(cancer_mondos)]

    print("Final CEPS shape after filter:", ceps.shape)
    print("Unique cancer MONDOs:", ceps["MONDO"].nunique())

    if not ceps.empty:
        print("Sample Disease Names:")
        print(ceps["DiseaseName"].unique()[:20])
        ceps.to_csv("merge.csv", index=False)
    else:
        print("WARNING: Dataset empty after cancer filter")

    return ceps


if __name__ == "__main__":
    df = merge()
    print("Merge completed.")
    print("Final dataset size:", df.shape)