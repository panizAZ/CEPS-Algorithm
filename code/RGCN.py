import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

graph_path = "hetero_graph.pt"

try:
    data = torch.load(graph_path).to(device)
    print(f"[INFO] Graph loaded successfully: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"[INFO] Edge types: {data.edge_type.unique()}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"{graph_path} not found. Run hetero_graph.py first to build and save the graph."
    )

embedding_dim = 256
hidden_dim = 256
out_dim = 64
rel_embedding_dim = 64
lr = 0.001
weight_decay = 1e-3
epochs = 100
weight_positive = 5.0
early_stopping_patience = 20
warmup_epochs = 10
dropout = 0.6

num_relations = int(data.edge_type.max()) + 1

node_emb = nn.Embedding(data.num_nodes, embedding_dim).to(device)
x = node_emb.weight

class RGCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, dropout=0.6):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv3 = RGCNConv(hidden_dim, out_dim, num_relations)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_type)
        return x

class DistMultDecoder(nn.Module):
    def __init__(self, num_relations, node_emb_dim, rel_emb_dim):
        super().__init__()
        self.relation_emb = nn.Embedding(num_relations, rel_emb_dim)
        self.proj = nn.Linear(node_emb_dim, rel_emb_dim)

    def forward(self, z, edge_index, edge_type):
        head, tail = edge_index[0], edge_index[1]
        h = self.proj(z[head])
        t = self.proj(z[tail])
        r = self.relation_emb(edge_type)
        return (h * r * t).sum(dim=-1)

encoder = RGCNEncoder(embedding_dim, hidden_dim, out_dim, num_relations, dropout).to(device)
decoder = DistMultDecoder(num_relations, out_dim, rel_embedding_dim).to(device)

all_edges = data.edge_index.cpu().numpy().T          
all_types = data.edge_type.cpu().numpy()           

train_edges_list = []
val_edges_list = []
train_types_list = []
val_types_list = []

for rel in range(num_relations):
    mask = all_types == rel
    rel_edges = all_edges[mask]
    if len(rel_edges) == 0:
        continue
    train, val = train_test_split(rel_edges, test_size=0.2, random_state=42)
    train_edges_list.append(train)
    val_edges_list.append(val)
    train_types_list.extend([rel] * len(train))
    val_types_list.extend([rel] * len(val))

train_edges = torch.tensor(np.concatenate(train_edges_list, axis=0), dtype=torch.long).t().to(device)
val_edges = torch.tensor(np.concatenate(val_edges_list, axis=0), dtype=torch.long).t().to(device)
train_types = torch.tensor(train_types_list, dtype=torch.long).to(device)
val_types = torch.tensor(val_types_list, dtype=torch.long).to(device)

edge_index_train = train_edges
edge_type_train = train_types

print(f"[INFO] Train edges: {train_edges.size(1)}")
print(f"[INFO] Val edges: {val_edges.size(1)}")

def relation_aware_negative_sampling(pos_edges, pos_types, num_nodes, num_neg_per_pos=3):

    pos_edges = pos_edges.t()  
    E = pos_edges.size(0)
    total_neg = E * num_neg_per_pos

    neg_edges = torch.zeros(total_neg, 2, dtype=torch.long, device=pos_edges.device)
    neg_types = torch.zeros(total_neg, dtype=torch.long, device=pos_types.device)

    for i in range(num_neg_per_pos):
        start = i * E
        end = (i + 1) * E
        neg_edges[start:end] = pos_edges.clone()
        neg_types[start:end] = pos_types

        corrupt_head = torch.randint(0, 2, (E,), device=pos_edges.device).bool()

        random_nodes = torch.randint(0, num_nodes, (E,), device=pos_edges.device)
 
        neg_edges[start:end, 0] = torch.where(corrupt_head, random_nodes, pos_edges[:, 0])
        neg_edges[start:end, 1] = torch.where(~corrupt_head, random_nodes, pos_edges[:, 1])

    return neg_edges.t().contiguous(), neg_types

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()) + list(node_emb.parameters()),
    lr=lr, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight_positive, device=device))

metrics = {"train_loss": [], "val_loss": [], "val_auc": []}
best_val_auc = 0
patience_counter = 0

print("\n[INFO] Starting training with relation-aware negative sampling...\n")

for epoch in range(1, epochs + 1):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()


    z = encoder(x, edge_index_train, edge_type_train)

    neg_edges, neg_types = relation_aware_negative_sampling(
        train_edges, train_types, data.num_nodes, num_neg_per_pos=3
    )


    pos_scores = decoder(z, train_edges, train_types)
    neg_scores = decoder(z, neg_edges, neg_types)

    
    y = torch.cat([
        torch.ones(pos_scores.size(0), device=device),
        torch.zeros(neg_scores.size(0), device=device)
    ])
    loss = criterion(torch.cat([pos_scores, neg_scores]), y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()) + list(node_emb.parameters()),
        max_norm=5.0
    )
    optimizer.step()

    
    if epoch == warmup_epochs:
        node_emb.requires_grad_(False)
        
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        print(f"[INFO] Node embeddings frozen after epoch {warmup_epochs}. Optimizer recreated.")

    
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        
        neg_val_edges, neg_val_types = relation_aware_negative_sampling(
            val_edges, val_types, data.num_nodes, num_neg_per_pos=3
        )

        val_pos_scores = decoder(z, val_edges, val_types)
        val_neg_scores = decoder(z, neg_val_edges, neg_val_types)

        val_y = torch.cat([
            torch.ones(val_pos_scores.size(0)),
            torch.zeros(val_neg_scores.size(0))
        ]).cpu().numpy()
        val_scores = torch.sigmoid(torch.cat([val_pos_scores, val_neg_scores])).cpu().numpy()
        val_loss = criterion(
            torch.cat([val_pos_scores, val_neg_scores]),
            torch.tensor(val_y, device=device)
        ).item()
        val_auc = roc_auc_score(val_y, val_scores)

    metrics["train_loss"].append(loss.item())
    metrics["val_loss"].append(val_loss)
    metrics["val_auc"].append(val_auc)

    scheduler.step(val_auc)

    print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0

        torch.save({
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "node_emb_state_dict": node_emb.state_dict(),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "rel_embedding_dim": rel_embedding_dim,
            "num_relations": num_relations,
            "decoder_type": "DistMult"
        }, "rgcn_distmult_best.pt")

        print("[INFO] Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

with open("rgcn_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nTraining complete.")
print(f"Best Val AUC: {best_val_auc:.4f}")