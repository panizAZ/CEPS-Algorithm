import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import networkx as nx
import plotly.graph_objects as go
import webbrowser


data = None
z = None
edge_index = None
edge_type = None
adj = None

chemical_nodes = None
gene_nodes = None
disease_nodes = None
offsets = None

all_paths = None
paths_by_pair = None

def initialize():

    global data, z, edge_index, edge_type, adj
    global chemical_nodes, gene_nodes, disease_nodes
    global offsets

    if z is not None:

        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing CEPS on: {device}")


    data = torch.load(
        "hetero_graph.pt",
        map_location=device,
        weights_only=False
    ).to(device)

    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_type = data.edge_type
    offsets = data.offsets

    chemical_nodes = list(range(offsets["chemical"], offsets["gene"]))
    gene_nodes     = list(range(offsets["gene"], offsets["disease"]))
    disease_nodes  = list(range(offsets["disease"], num_nodes))

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

    checkpoint = torch.load(
        "rgcn_distmult_best.pt",
        map_location=device,
        weights_only=False
    )

    embedding_dim = checkpoint["embedding_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    out_dim = checkpoint["out_dim"]
    num_relations = checkpoint["num_relations"]

    node_emb = nn.Embedding(num_nodes, embedding_dim).to(device)
    node_emb.load_state_dict(checkpoint["node_emb_state_dict"])

    encoder = RGCNEncoder(
        embedding_dim, hidden_dim, out_dim, num_relations
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    with torch.no_grad():
        z = encoder(node_emb.weight, edge_index, edge_type).detach()

    adj = {n: set() for n in range(num_nodes)}
    for u, v in edge_index.t().tolist():
        adj[u].add(v)
        adj[v].add(u)

    print("[INFO] CEPS initialized ✅")



def link_score(u, v, z_override=None, return_tensor=False):

    emb = z_override if z_override is not None else z

    sim = torch.cosine_similarity(
        emb[u].unsqueeze(0),
        emb[v].unsqueeze(0)
    )
    score = (sim + 1) / 2  

    if return_tensor:
        return score.squeeze()
    else:
        return score.item()

def visualize_paths(all_paths, paths_by_pair, filename="explained_paths_interactive.html"):

    initialize()

    G = nx.Graph()

    global_top_paths = all_paths
    for score, path in global_top_paths:
        c, g, d = path
        G.add_node(c, type="chemical", label=data.idx2chemical[c - offsets["chemical"]])
        G.add_node(g, type="gene", label=data.idx2gene[g - offsets["gene"]])
        G.add_node(d, type="disease", label=data.idx2disease[d - offsets["disease"]])


    nodes_by_type = {"chemical": [], "gene": [], "disease": []}
    for n, d in G.nodes(data=True):
        nodes_by_type[d["type"]].append(n)

    max_nodes = max(len(nodes_by_type["chemical"]),
                    len(nodes_by_type["gene"]),
                    len(nodes_by_type["disease"]))
    y_gap = max(1500, 30000 // max_nodes if max_nodes > 0 else 1500)
    x_shift = {"chemical": 0, "gene": 120, "disease": 240}

    pos = {}
    for i, n in enumerate(nodes_by_type["chemical"]):
        pos[n] = (x_shift["chemical"], -i * y_gap)
    for i, n in enumerate(nodes_by_type["gene"]):
        pos[n] = (x_shift["gene"], -i * y_gap)
    for i, n in enumerate(nodes_by_type["disease"]):
        pos[n] = (x_shift["disease"], -i * y_gap)

    edge_traces_global = []
    for global_rank, (score, path) in enumerate(global_top_paths):
        c, g, d = path
        color = "red" if global_rank < 3 else "rgba(180,180,180,0.25)"
        text_color = "darkred" if global_rank < 3 else "gray"
        for u, v in [(c, g), (g, d)]:
            x0, y0 = pos[u];
            x1, y1 = pos[v]
            xm = (x0 + x1) / 2;
            ym = (y0 + y1) / 2
            edge_traces_global.append(
                go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                           line=dict(width=0.8 + 3 * score, color=color),
                           hoverinfo="skip", showlegend=False)
            )
            edge_traces_global.append(
                go.Scatter(x=[xm], y=[ym + 20], mode="text", text=[f"{score:.3f}"],
                           textfont=dict(size=12, color=text_color),
                           showlegend=False, hoverinfo="skip")
            )

    edge_traces_local = []
    trace_meta_local = []
    TOP_K_PER_PAIR = 5

    for (c, d), paths in paths_by_pair.items():
        if c not in G.nodes() or d not in G.nodes():
            continue
        for score, path, global_rank, local_rank in paths[:TOP_K_PER_PAIR]:
            _, g, _ = path
            if g not in G.nodes():
                continue
            color = "red" if local_rank < 3 else "rgba(180,180,180,0.25)"
            text_color = "darkred" if local_rank < 3 else "gray"
            for u, v in [(c, g), (g, d)]:
                x0, y0 = pos[u];
                x1, y1 = pos[v]
                xm = (x0 + x1) / 2;
                ym = (y0 + y1) / 2
                edge_traces_local.append(
                    go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                               line=dict(width=0.8 + 3 * score, color=color),
                               hoverinfo="skip", showlegend=False)
                )
                trace_meta_local.append((c, d))
                edge_traces_local.append(
                    go.Scatter(x=[xm], y=[ym + 20], mode="text", text=[f"{score:.3f}"],
                               textfont=dict(size=12, color=text_color),
                               showlegend=False, hoverinfo="skip")
                )
                trace_meta_local.append((c, d))

    def node_trace_for_type(node_type, color, size):
        xs, ys, texts, hovers = [], [], [], []
        for n, d in G.nodes(data=True):
            if d["type"] == node_type:
                x, y = pos[n]
                xs.append(x);
                ys.append(y)
                label = d["label"]
                hover = f"<b>{label}</b><br>Type: {node_type}<br>ID: {n}"
                texts.append(label)
                hovers.append(hover)
        return go.Scatter(
            x=xs, y=ys, mode="markers+text", text=texts,
            textposition="middle right", hoverinfo="text", hovertext=hovers,
            marker=dict(size=size, color=color, line=dict(width=1.5, color="black")),
            showlegend=True, name=node_type.capitalize()
        )

    node_traces = [
        node_trace_for_type("chemical", "#8dd3c7", 20),
        node_trace_for_type("gene", "#ffffb3", 16),
        node_trace_for_type("disease", "#fb8072", 20),
    ]

    buttons = []

    initial_visibility = [False] * len(edge_traces_global) + [False] * len(edge_traces_local) + [True] * len(
        node_traces)

    overview_visibility = [True] * len(edge_traces_global) + [False] * len(edge_traces_local) + [True] * len(
        node_traces)
    buttons.append(dict(
        label="Overview",
        method="update",
        args=[{"visible": overview_visibility}, {"title": "CEPS Explanation: Overview"}]
    ))

    valid_pairs_local = sorted(set(trace_meta_local))
    for (c, d) in valid_pairs_local:
        visibility = [False] * len(edge_traces_global) + \
                     [(meta == (c, d)) for meta in trace_meta_local] + \
                     [True] * len(node_traces)
        label = f"{data.idx2chemical[c - offsets['chemical']]}, {data.idx2disease[d - offsets['disease']]}"
        buttons.append(dict(
            label=label,
            method="update",
            args=[{"visible": visibility}, {"title": f"CEPS Explanation: {label}"}]
        ))

    fig = go.Figure(
        data=edge_traces_global + edge_traces_local + node_traces,
        layout=go.Layout(
            title=dict(text="Interactive CEPS Explanation", font=dict(size=20), y=0.98),
            hovermode="closest",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=40, r=40, t=120, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                x=0.01,
                y=1.12,
                showactive=True,
                bgcolor="white",
                bordercolor="#ccc",
                font=dict(size=11)
            )]
        )
    )

    for i in range(len(edge_traces_global) + len(edge_traces_local)):
        fig.data[i].visible = False

    fig.update_layout(autosize=True, margin=dict(l=80, r=80, t=120, b=80))

    config = {
        'responsive': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'],
        'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'resetScale2d', 'toimage']
    }

    html_str = fig.to_html(include_plotlyjs=True, config=config)
    css = """
    <style>
    .modebar-btn {
        font-size: 20px !important;
        width: 32px !important;
        height: 32px !important;
    }
    .modebar-btn .icon {
        width: 20px !important;
        height: 20px !important;
    }
    </style>
    """
    html_str = html_str.replace('</head>', css + '</head>')

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_str)

    webbrowser.open(filename)

def rank_pairs(paths_by_pair, mode="best"):

    pair_scores = []

    for (c, d), paths in paths_by_pair.items():
        scores = [score for score, *_ in paths]

        if mode == "best":
            s = max(scores)
        elif mode == "mean":
            s = sum(scores) / len(scores)
        elif mode == "sum":
            s = sum(scores)
        else:
            raise ValueError("Unknown mode")

        pair_scores.append((s, (c, d)))

    pair_scores.sort(reverse=True, key=lambda x: x[0])
    return pair_scores



if __name__ == "__main__":
    initialize()