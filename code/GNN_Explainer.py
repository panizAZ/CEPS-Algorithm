import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] GNN Explainer running on: {device}")



def explain_link_edge_aware(
    z,
    edge_index,
    src,
    dst,
    link_score,
    epochs=300,
    lr=0.1,
    lam_sparse=0.01,
    lam_entropy=0.01,
    verbose=True
):


    device = z.device
    num_edges = edge_index.size(1)

    edge_mask_param = nn.Parameter(
        torch.randn(num_edges, device=device)
    )

    optimizer = torch.optim.Adam([edge_mask_param], lr=lr)

    with torch.no_grad():
        base_score = link_score(src, dst)

    if verbose:
        print(f"[INFO] Base link score: {base_score:.4f}")

    for epoch in range(epochs):
        mask = torch.sigmoid(edge_mask_param)


        node_weight = torch.zeros(z.size(0), device=device)
        node_weight.index_add_(0, edge_index[0], mask)
        node_weight.index_add_(0, edge_index[1], mask)

        node_weight = node_weight / (node_weight.max() + 1e-8)
        z_masked = z * node_weight.unsqueeze(1)

        pred = link_score(src, dst, z_override=z_masked, return_tensor=True)

        loss_pred = (pred - base_score) ** 2
        loss_sparse = mask.mean()
        loss_entropy = (-mask * torch.log(mask + 1e-8)).mean()

        loss = loss_pred + lam_sparse * loss_sparse + lam_entropy * loss_entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and epoch % 50 == 0:
            print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} pred={pred.item():.4f}")

    return torch.sigmoid(edge_mask_param).detach()



def important_nodes_from_edge_mask(edge_mask, edge_index, ratio=0.4):

    num_nodes = int(edge_index.max()) + 1
    node_score = torch.zeros(num_nodes, device=edge_mask.device)

    node_score.index_add_(0, edge_index[0], edge_mask)
    node_score.index_add_(0, edge_index[1], edge_mask)

    k = max(1, int(num_nodes * ratio))
    return set(node_score.topk(k).indices.tolist())


def enumerate_ceps_paths_masked(
    important_nodes,
    adj,
    chemical_nodes,
    gene_nodes,
    disease_nodes,
    link_score
):
    paths = []

    for c in chemical_nodes:
        if c not in important_nodes:
            continue

        for g in gene_nodes:
            if g not in important_nodes or g not in adj[c]:
                continue

            for d in disease_nodes:
                if d not in important_nodes or d not in adj[g]:
                    continue

                score = link_score(c, g) + link_score(g, d)
                paths.append((score, [c, g, d]))

    paths.sort(reverse=True, key=lambda x: x[0])
    return paths


def explain_and_enumerate_paths(
    z,
    edge_index,
    src,
    dst,
    adj,
    chemical_nodes,
    gene_nodes,
    disease_nodes,
    link_score,
    node_ratio=0.4,
    epochs=300
):
    print("[INFO] Running GNN Explainer (edge‑aware)...")

    edge_mask = explain_link_edge_aware(
        z=z,
        edge_index=edge_index,
        src=src,
        dst=dst,
        link_score=link_score,
        epochs=epochs
    )

    important_nodes = important_nodes_from_edge_mask(
        edge_mask, edge_index, ratio=node_ratio
    )

    print(f"[INFO] Important nodes selected: {len(important_nodes)}")

    print("[INFO] Enumerating CEPS-style paths...")
    paths = enumerate_ceps_paths_masked(
        important_nodes,
        adj,
        chemical_nodes,
        gene_nodes,
        disease_nodes,
        link_score
    )

    print(f"[INFO] Total paths found: {len(paths)}")

    return {
        "edge_mask": edge_mask,
        "important_nodes": important_nodes,
        "paths": paths
    }