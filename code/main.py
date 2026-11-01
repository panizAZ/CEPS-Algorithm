from GNN_Explainer import explain_and_enumerate_paths
import CEPS
import csv
import torch
import index_lookup



TOP_K_PAIRS = 20       
TOP_M_PATHS = 5        

SPECIFIC_CHEMICAL = "Arsenic"
SPECIFIC_DISEASE = "Osteosarcoma"



def main():

    CEPS.initialize()


    src = CEPS.chemical_nodes[0]
    dst = CEPS.disease_nodes[0]

    print(
        "[INFO] Explaining:",
        CEPS.data.idx2chemical[src - CEPS.offsets["chemical"]],
        "→",
        CEPS.data.idx2disease[dst - CEPS.offsets["disease"]],
    )

    result = explain_and_enumerate_paths(
        z=CEPS.z,
        edge_index=CEPS.edge_index,
        src=src,
        dst=dst,
        adj=CEPS.adj,
        chemical_nodes=CEPS.chemical_nodes,
        gene_nodes=CEPS.gene_nodes,
        disease_nodes=CEPS.disease_nodes,
        link_score=CEPS.link_score,
        node_ratio=0.4,
        epochs=150,
    )

    paths = result["paths"]
    print(f"[INFO] Paths found after explanation: {len(paths)}")

    torch.save(
        {
            "src": src,
            "dst": dst,
            "paths": paths,
            "important_nodes": result["important_nodes"],
        },
        "explanation_result.pt",
    )

    with open("explained_paths.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["score", "chemical", "gene", "disease"])
        for score, (c, g, d) in paths:
            writer.writerow(
                [
                    score,
                    CEPS.data.idx2chemical[c - CEPS.offsets["chemical"]],
                    CEPS.data.idx2gene[g - CEPS.offsets["gene"]],
                    CEPS.data.idx2disease[d - CEPS.offsets["disease"]],
                ]
            )

    print("[INFO] Explained paths saved ✅")


    CEPS.all_paths = paths
    CEPS.paths_by_pair = {}

    for global_rank, (score, path) in enumerate(paths):
        c, g, d = path
        pair = (c, d)
        CEPS.paths_by_pair.setdefault(pair, []).append(
            (score, path, global_rank)
        )

    for pair in CEPS.paths_by_pair:
        CEPS.paths_by_pair[pair].sort(reverse=True, key=lambda x: x[0])
        CEPS.paths_by_pair[pair] = [
            (score, path, global_rank, local_rank)
            for local_rank, (score, path, global_rank)
            in enumerate(CEPS.paths_by_pair[pair])
        ]

    print("[INFO] Explained pairs:", len(CEPS.paths_by_pair))
    print("[INFO] Explained paths:", len(paths))


    if SPECIFIC_CHEMICAL and SPECIFIC_DISEASE:
        c = index_lookup.get_chemical_index(SPECIFIC_CHEMICAL)
        d = index_lookup.get_disease_index(SPECIFIC_DISEASE)

        pair = (c, d)
        print(
            "[INFO] Visualizing specific pair:",
            CEPS.data.idx2chemical[c - CEPS.offsets["chemical"]],
            "→",
            CEPS.data.idx2disease[d - CEPS.offsets["disease"]],
        )

        if pair not in CEPS.paths_by_pair:
            print("[WARN] No explainer-valid paths for this pair.")
        else:
            pair_paths = CEPS.paths_by_pair[pair][:TOP_M_PATHS]

            pair_paths_by_pair = {pair: pair_paths}
            pair_all_paths = [(score, path) for score, path, _, _ in pair_paths]

            CEPS.visualize_paths(
                all_paths=pair_all_paths,
                paths_by_pair=pair_paths_by_pair,
                filename="pair_specific_paths.html",
            )


    ranked_pairs = CEPS.rank_pairs(CEPS.paths_by_pair, mode="best")
    selected_pairs = [pair for _, pair in ranked_pairs[:TOP_K_PAIRS]]

    filtered_paths_by_pair = {}
    filtered_all_paths = []

    for pair in selected_pairs:
        top_paths = CEPS.paths_by_pair[pair][:TOP_M_PATHS]
        filtered_paths_by_pair[pair] = top_paths
        for score, path, _, _ in top_paths:
            filtered_all_paths.append((score, path))

    print(
        f"[INFO] Visualizing {len(filtered_paths_by_pair)} pairs "
        f"({TOP_M_PATHS} paths per pair)"
    )

    CEPS.visualize_paths(
        all_paths=filtered_all_paths,
        paths_by_pair=filtered_paths_by_pair,
        filename="top_pairs_paths.html",
    )


if __name__ == "__main__":
    main()