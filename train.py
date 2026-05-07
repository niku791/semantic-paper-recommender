import torch
import torch.nn.functional as F
from torch import nn

from config import params
from data_utils import fetch_papers, build_meta_path_adjs
from models.aggregator import SemanticAggregator, aggregate_meta_path
from models.fusion import SemanticFusion


def load_dataset():
    works = fetch_papers(query=None, limit=params["subset_size"])
    print("Loaded papers:", len(works))
    return works


def load_embeddings(n, device):
    print("Loading precomputed embeddings...")
    paper_emb = torch.load("paper_embeddings.pt", map_location=device)
    paper_emb = paper_emb[:n]
    print("Embedding shape:", paper_emb.shape)
    return paper_emb


def contrastive_loss(fused_items, all_neighbors, device, temperature=0.1):
    n = fused_items.shape[0]
    sim = torch.matmul(fused_items, fused_items.T) / temperature
    total_loss = torch.tensor(0.0, device=device)
    count = 0
    for i in range(n):
        pos_idx = list(all_neighbors.get(i, set()) - {i})
        if len(pos_idx) == 0:
            continue
        pos_idx = torch.tensor(pos_idx, dtype=torch.long, device=device)
        log_probs = F.log_softmax(sim[i], dim=0)
        total_loss = total_loss - log_probs[pos_idx].mean()
        count += 1
    return total_loss / max(count, 1)


def train_model(query=None):
    device = params["device"]
    works = load_dataset()
    paper_emb = load_embeddings(len(works), device)

    (pap_neighbors, pvp_neighbors,
     pyp_neighbors, pkp_neighbors,
     pcp_neighbors) = build_meta_path_adjs(works)

    all_neighbors = {}
    for i in range(len(works)):
        all_neighbors[i] = (
            pap_neighbors.get(i, set()) |
            pvp_neighbors.get(i, set()) |
            pyp_neighbors.get(i, set()) |
            pkp_neighbors.get(i, set()) |
            pcp_neighbors.get(i, set())
        )

    ag_pap = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pvp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pyp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pkp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pcp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)

    fused_dim = params["semantic_proj_dim"] * params["L"]
    proj_identity = nn.Linear(params["st_dim"], fused_dim).to(device)
    fusion = SemanticFusion(fused_dim, params["attn_dim"]).to(device)

    optimizer = torch.optim.Adam(
        list(ag_pap.parameters()) +
        list(ag_pvp.parameters()) +
        list(ag_pyp.parameters()) +
        list(ag_pkp.parameters()) +
        list(ag_pcp.parameters()) +
        list(proj_identity.parameters()) +
        list(fusion.parameters()),
        lr=params["lr"]
    )

    for epoch in range(params["epochs"]):
        optimizer.zero_grad()

        E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, ag_pap, L=params["L"])
        E_pvp = aggregate_meta_path(works, paper_emb, pvp_neighbors, ag_pvp, L=params["L"])
        E_pyp = aggregate_meta_path(works, paper_emb, pyp_neighbors, ag_pyp, L=params["L"])
        E_pkp = aggregate_meta_path(works, paper_emb, pkp_neighbors, ag_pkp, L=params["L"])
        E_pcp = aggregate_meta_path(works, paper_emb, pcp_neighbors, ag_pcp, L=params["L"])

        E_identity = torch.relu(proj_identity(paper_emb))

        fused_items, weights = fusion([E_identity, E_pap, E_pvp, E_pyp, E_pkp, E_pcp])
        fused_items = F.normalize(fused_items, dim=1)

        loss = contrastive_loss(fused_items, all_neighbors, device)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{params['epochs']}  Loss: {loss.item():.4f}")

    torch.save({
        "aggregator_pap": ag_pap.state_dict(),
        "aggregator_pvp": ag_pvp.state_dict(),
        "aggregator_pyp": ag_pyp.state_dict(),
        "aggregator_pkp": ag_pkp.state_dict(),
        "aggregator_pcp": ag_pcp.state_dict(),
        "proj_identity":  proj_identity.state_dict(),
        "fusion":         fusion.state_dict(),
    }, "model_trained.pt")

    print("Model saved → model_trained.pt")


if __name__ == "__main__":
    train_model()
