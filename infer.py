import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer

from config import params
from data_utils import fetch_papers, build_meta_path_adjs
from models.aggregator import SemanticAggregator, aggregate_meta_path
from models.fusion import SemanticFusion


def load_model(device):
    fused_dim = params["semantic_proj_dim"] * params["L"]

    ag_pap = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pvp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pyp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pkp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    ag_pcp = SemanticAggregator(params["st_dim"], params["semantic_proj_dim"]).to(device)
    proj_identity = nn.Linear(params["st_dim"], fused_dim).to(device)
    fusion = SemanticFusion(fused_dim, params["attn_dim"]).to(device)

    ckpt = torch.load("model_trained.pt", map_location=device)
    ag_pap.load_state_dict(ckpt["aggregator_pap"])
    ag_pvp.load_state_dict(ckpt["aggregator_pvp"])
    ag_pyp.load_state_dict(ckpt["aggregator_pyp"])
    ag_pkp.load_state_dict(ckpt["aggregator_pkp"])
    ag_pcp.load_state_dict(ckpt["aggregator_pcp"])
    proj_identity.load_state_dict(ckpt["proj_identity"])
    fusion.load_state_dict(ckpt["fusion"])

    return ag_pap, ag_pvp, ag_pyp, ag_pkp, ag_pcp, proj_identity, fusion


def recommend(query, limit=25, top_k=10):
    device = params["device"]

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    works = fetch_papers(query, limit=limit)
    if not works:
        return []

    titles = [w.get("title", "") or "" for w in works]
    paper_emb = torch.tensor(
        st_model.encode(titles), dtype=torch.float, device=device
    )

    (pap_neighbors, pvp_neighbors,
     pyp_neighbors, pkp_neighbors,
     pcp_neighbors) = build_meta_path_adjs(works)

    ag_pap, ag_pvp, ag_pyp, ag_pkp, ag_pcp, proj_identity, fusion = load_model(device)

    E_pap = aggregate_meta_path(works, paper_emb, pap_neighbors, ag_pap, L=params["L"])
    E_pvp = aggregate_meta_path(works, paper_emb, pvp_neighbors, ag_pvp, L=params["L"])
    E_pyp = aggregate_meta_path(works, paper_emb, pyp_neighbors, ag_pyp, L=params["L"])
    E_pkp = aggregate_meta_path(works, paper_emb, pkp_neighbors, ag_pkp, L=params["L"])
    E_pcp = aggregate_meta_path(works, paper_emb, pcp_neighbors, ag_pcp, L=params["L"])
    E_identity = torch.relu(proj_identity(paper_emb))

    fused_items, meta_weights = fusion([E_identity, E_pap, E_pvp, E_pyp, E_pkp, E_pcp])
    fused_items = F.normalize(fused_items, dim=1)

    q_emb = torch.tensor(
        st_model.encode([query])[0], dtype=torch.float, device=device
    )
    q_proj = F.normalize(proj_identity(q_emb), dim=0)

    scores = torch.mv(fused_items, q_proj)
    k = min(top_k, len(scores))
    topk = torch.topk(scores, k=k)

    recs = []
    for rank, (idx, score) in enumerate(
        zip(topk.indices.tolist(), topk.values.tolist()), start=1
    ):
        p = works[idx]
        recs.append({
            "rank":    rank,
            "title":   p.get("title", "N/A"),
            "authors": [a.get("name") for a in p.get("authors", [])][:5],
            "venue":   p.get("venue", "N/A"),
            "year":    p.get("year", "N/A"),
            "url":     p.get("url", "N/A"),
            "score":   float(score),
        })

    return recs, meta_weights
