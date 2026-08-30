# -*- coding: utf-8 -*-
# Extracted from: Chapter 113 — GraphRAG and Knowledge-Graph Retrieval
# Source: src/.../ch113-graphrag.md   Tier: A
# Regenerate with: make code  (do not edit by hand)

"""Multi-hop questions: what a graph actually buys, and what it charges.

A two-hop question -- "which supplier serves the depot that handles the Lyon
account" -- names X and Z, and the answer needs the chunk linking X to Y and the
chunk linking Y to Z. ch:rag-query-understanding showed why one-shot retrieval
struggles: the second hop's search terms do not exist until the first hop has
been answered.

A graph replaces search with traversal, and traversal succeeds only if EVERY edge
on the path was extracted correctly (eq:path-reliability). This listing measures
both sides on the same corpus: vector retrieval's success as a function of entity
degree, and graph traversal's as a function of per-edge extraction accuracy --
plus the neighbourhood explosion that is traversal's precision cost.
"""
import numpy as np

rng = np.random.default_rng(23)

N_ENT = 800
N_QUERY = 400
FP_RATE = 0.10          # spurious edges, as a fraction of true edges


def build_corpus(degree):
    """Each true relation is stated in exactly one chunk, which mentions its two
    endpoints. A chunk is therefore an edge, which is what makes the comparison
    between retrieval and traversal a fair one."""
    edges = set()
    while len(edges) < N_ENT * degree // 2:
        a, b = rng.integers(0, N_ENT, size=2)
        if a != b:
            edges.add((min(a, b), max(a, b)))
    edges = np.array(sorted(edges))
    inc = [[] for _ in range(N_ENT)]          # entity -> chunk ids mentioning it
    adj = [set() for _ in range(N_ENT)]
    for i, (a, b) in enumerate(edges):
        inc[a].append(i)
        inc[b].append(i)
        adj[a].add(b)
        adj[b].add(a)
    return edges, inc, adj


def sample_path(adj, hops):
    """A path X -> ... -> Z with no shortcut edge, so the question genuinely
    needs every hop."""
    for _ in range(200):
        path = [int(rng.integers(0, N_ENT))]
        ok = True
        for _ in range(hops):
            nxt = list(adj[path[-1]] - set(path))
            if not nxt:
                ok = False
                break
            path.append(nxt[int(rng.integers(0, len(nxt)))])
        if ok and len(path) == hops + 1 and path[-1] not in adj[path[0]]:
            return path
    return None


def edge_id(edges_lookup, a, b):
    return edges_lookup.get((min(a, b), max(a, b)))


def vector_recall(inc, adj, lookup, hops, k):
    """One-shot retrieval. The query names X and Z, so every chunk mentioning
    either is a candidate and nothing distinguishes the ones on the path.
    Success means every path chunk lands in the top k (eq:hub-dilution)."""
    hit = 0
    for _ in range(N_QUERY):
        path = sample_path(adj, hops)
        if path is None:
            continue
        x, z = path[0], path[-1]
        cand = sorted(set(inc[x]) | set(inc[z]))
        # Chunks mentioning a query entity are indistinguishable to the ranker;
        # noise decides the order among them.
        order = [cand[i] for i in np.argsort(rng.random(len(cand)))][:k]
        need = {edge_id(lookup, path[i], path[i + 1]) for i in range(hops)}
        hit += int(need <= set(order))
    return hit / N_QUERY


def graph_recall(edges, lookup, adj, hops, p_e):
    """Traversal over an EXTRACTED graph: each true edge survives extraction with
    probability p_e, and spurious edges are added at FP_RATE. The path is
    recovered only if every edge on it survived."""
    keep = rng.random(len(edges)) < p_e
    ext = [set() for _ in range(N_ENT)]
    for i, (a, b) in enumerate(edges):
        if keep[i]:
            ext[a].add(b)
            ext[b].add(a)
    for _ in range(int(FP_RATE * len(edges))):
        a, b = rng.integers(0, N_ENT, size=2)
        if a != b:
            ext[int(a)].add(int(b))
            ext[int(b)].add(int(a))

    hit, frontier_total, n = 0, 0, 0
    for _ in range(N_QUERY):
        path = sample_path(adj, hops)
        if path is None:
            continue
        hit += int(all(path[i + 1] in ext[path[i]] for i in range(hops)))
        seen, frontier = {path[0]}, {path[0]}
        for _ in range(hops):
            frontier = set().union(*(ext[v] for v in frontier)) - seen if frontier else set()
            seen |= frontier
        frontier_total += len(seen)
        n += 1
    return hit / max(n, 1), frontier_total / max(n, 1)


print(f"{N_ENT} entities; every relation stated in exactly one chunk; "
      f"{FP_RATE:.0%} spurious extracted edges\n")

print("ONE-SHOT VECTOR RETRIEVAL -- success by entity degree and budget k")
print(f"{'degree':>8}{'chunks':>9}{'k=10':>9}{'k=25':>9}{'k=50':>9}{'k=100':>9}")
print("-" * 53)
vec = {}
for degree in (4, 8, 20, 50):
    edges_g, inc_g, adj_g = build_corpus(degree)
    lookup_g = {(int(a), int(b)): i for i, (a, b) in enumerate(edges_g)}
    row = [vector_recall(inc_g, adj_g, lookup_g, 2, k) for k in (10, 25, 50, 100)]
    vec[degree] = row
    print(f"{degree:>8}{len(edges_g):>9}" + "".join(f"{v:>9.3f}" for v in row))

print("\nGRAPH TRAVERSAL -- success by per-edge extraction accuracy, and the")
print("size of the neighbourhood the traversal returns (degree 8)")
edges_g, inc_g, adj_g = build_corpus(8)
lookup_g = {(int(a), int(b)): i for i, (a, b) in enumerate(edges_g)}
print(f"{'p_edge':>8}{'2-hop':>9}{'|N_2|':>9}{'3-hop':>9}{'|N_3|':>9}{'4-hop':>9}")
print("-" * 53)
gr = {}
for p_e in (0.60, 0.75, 0.85, 0.95, 1.00):
    r2, n2 = graph_recall(edges_g, lookup_g, adj_g, 2, p_e)
    r3, n3 = graph_recall(edges_g, lookup_g, adj_g, 3, p_e)
    r4, _ = graph_recall(edges_g, lookup_g, adj_g, 4, p_e)
    gr[p_e] = (r2, n2, r3, n3, r4)
    print(f"{p_e:>8.2f}{r2:>9.3f}{n2:>9.0f}{r3:>9.3f}{n3:>9.0f}{r4:>9.3f}")

print(f"""
Read the first table before the second. At degree 4 one-shot vector retrieval
answers two-hop questions well ({vec[4][1]:.3f} at k=25) -- the entities are
mentioned in few chunks, so a generous k simply retrieves all of them and the
path is inside the retrieved set. The widely repeated claim that vector retrieval
cannot do multi-hop is false at low degree, and small benchmark corpora are
exactly where degree is low.

Degree is what breaks it. At degree 50 the same k=25 scores {vec[50][1]:.3f},
because the two query entities are mentioned in about a hundred chunks and
nothing in the ranking distinguishes the two that matter. This is the honest
statement of the problem a graph solves: not "multi-hop", but HUB ENTITIES, where
the number of chunks mentioning the query terms exceeds the budget
(eq:hub-dilution).

The second table prices the alternative. Traversal is exact when extraction is
exact -- the p_edge = 1.00 row is a perfect {gr[1.00][0]:.3f} at every depth --
and extraction is never exact. At p_edge = 0.85, which is a good entity-and-
relation extractor, two-hop success is {gr[0.85][0]:.3f} and four-hop is
{gr[0.85][4]:.3f}: each additional hop multiplies by p_edge again
(eq:path-reliability), so depth is bought with reliability at a compounding rate.

Compare the two tables at the setting where the graph is supposed to win. At
degree 50 and k=25 retrieval scores {vec[50][1]:.3f} and traversal at p_edge=0.85
scores {gr[0.85][0]:.3f}, so the graph is decisively right. At degree 4 retrieval
scores {vec[4][1]:.3f} and the same graph still scores {gr[0.85][0]:.3f}, so the
graph is decisively WRONG -- it has spent a whole-corpus extraction pass to make
a solved problem worse. Nothing about the technique changed between those two
rows. The corpus did.

And the neighbourhood column is the cost nobody quotes. Traversal does not return
an answer, it returns everything within h hops: {gr[0.85][1]:.0f} entities at two
hops and {gr[0.85][3]:.0f} at three, from a graph of {N_ENT}. By the third hop
the traversal has touched almost half the corpus, so the graph has converted a
precision problem into a recall problem and handed the precision problem back to
whatever reranks the result (ch:emb-reranking).

One detail worth noticing before trusting these numbers: two-hop success at
p_edge = 0.85 is {gr[0.85][0]:.3f}, slightly ABOVE the {0.85 ** 2:.3f} that
eq:path-reliability predicts. The excess is spurious edges accidentally
reconnecting a pair that extraction dropped -- the traversal is right for the
wrong reason. It is a small effect here and a warning in general: a graph with a
false-positive rate can answer correctly by coincidence, and a system evaluated
only on answers will not notice.""")
