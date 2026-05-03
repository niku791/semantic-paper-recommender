import json
import os
from collections import defaultdict


def reconstruct_abstract(inv_index):
    if not inv_index or not isinstance(inv_index, dict):
        return ""
    words = []
    for word, positions in inv_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort()
    return " ".join([w for _, w in words])


def fetch_papers(query=None, limit=50, dataset_path="papers.json", embeddings_path="paper_embeddings.pt"):
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset file papers.json not found")

    with open(dataset_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if query is None:
        return papers[:limit]

    # Dense retrieval — encode query and compare with precomputed embeddings
    if os.path.exists(embeddings_path):
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        q_emb = torch.tensor(model.encode([query])[0], dtype=torch.float)
        q_emb = F.normalize(q_emb.unsqueeze(0), dim=1)

        paper_emb = torch.load(embeddings_path, map_location="cpu")
        paper_emb = F.normalize(paper_emb, dim=1)

        scores = torch.mv(paper_emb, q_emb.squeeze())
        top_indices = torch.topk(scores, k=min(limit, len(papers))).indices.tolist()
        return [papers[i] for i in top_indices]

    # Fallback to keyword search if embeddings not available
    query = query.lower()
    results = []
    for p in papers:
        title = (p.get("title") or "").lower()
        abstract = reconstruct_abstract(p.get("abstract")).lower()
        if query in title or query in abstract:
            results.append(p)
        if len(results) >= limit:
            break
    return results


def build_meta_path_adjs(works):

    author_to_papers   = defaultdict(list)
    venue_to_papers    = defaultdict(list)
    year_to_papers     = defaultdict(list)
    keyword_to_papers  = defaultdict(list)
    id_to_index        = {}

    for i, w in enumerate(works):
        pid = w.get("paperId")
        if pid:
            id_to_index[pid] = i

    for i, w in enumerate(works):
        for a in w.get("authors", []):
            name = a.get("name")
            if name:
                author_to_papers[name].append(i)
        venue = w.get("venue")
        if venue:
            venue_to_papers[venue].append(i)
        year = w.get("year")
        if year:
            year_to_papers[year].append(i)
        for c in w.get("concepts", []):
            keyword_to_papers[c].append(i)

    pap_neighbors = defaultdict(set)
    pvp_neighbors = defaultdict(set)
    pyp_neighbors = defaultdict(set)
    pkp_neighbors = defaultdict(set)
    pcp_neighbors = defaultdict(set)

    for papers in author_to_papers.values():
        for p in papers:
            pap_neighbors[p].update(papers)

    for papers in venue_to_papers.values():
        for p in papers:
            pvp_neighbors[p].update(papers)

    for papers in year_to_papers.values():
        for p in papers:
            pyp_neighbors[p].update(papers)

    for papers in keyword_to_papers.values():
        for p in papers:
            pkp_neighbors[p].update(papers)

    for i, w in enumerate(works):
        for ref in w.get("references", []):
            if ref in id_to_index:
                j = id_to_index[ref]
                pcp_neighbors[i].add(j)
                pcp_neighbors[j].add(i)

    for p in range(len(works)):
        pap_neighbors[p].discard(p)
        pvp_neighbors[p].discard(p)
        pyp_neighbors[p].discard(p)
        pkp_neighbors[p].discard(p)
        pcp_neighbors[p].discard(p)

    return (
        pap_neighbors,
        pvp_neighbors,
        pyp_neighbors,
        pkp_neighbors,
        pcp_neighbors
    )
