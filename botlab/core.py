"""
botlab.core  --  working prototype of the research proposal
"Fake Account and Bot Network Detection via Graph Structure".

Pipeline (mirrors Figure 1 / Section 5 of the proposal):
  1. Data preparation   -> synthetic NetworkX graph with planted bot clusters + simulated timestamps
  2. Feature engineering-> structural + temporal features (Section 6)
  3. Models             -> LogReg, RandomForest, classical anomaly detectors, GraphSAGE, GAT,
                           and a SEPARATE text-only content baseline
  4. Robustness testing -> controlled changes in bot behaviour (camouflage, sparse groups, ...)
  5. Evaluation         -> Precision / Recall / F1 / PR-AUC / ROC-AUC / dF1 / group-level / time
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace, asdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# 0. Configuration
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    n_genuine: int = 450
    n_bots: int = 90
    group_size: int = 15          # bots are planted in coordinated groups of this size
    camouflage: float = 0.0       # 0..1  bots befriend genuine users / look normal
    group_density: float = 0.35   # 0..1  edge probability inside a bot group (sparse = low)
    sync_spread: float = 3.0      # std-dev (days) of account-creation dates inside a group
    timing_noise: float = 0.15    # 0 = clock-like cadence ... 1+ = random human-like timing
    llm_text: bool = False        # bots post LLM-style text that imitates genuine users
    seed: int = 7


SCENARIOS = {
    "Baseline (no change)": {},
    "Bots befriend genuine users (camouflage)": {"camouflage": 0.85},
    "Sparse bot groups": {"group_density": 0.06},
    "Reduced synchronization": {"sync_spread": 250.0},
    "Changed activity timing": {"timing_noise": 1.3},
    "AI-generated text (LLM bots)": {"llm_text": True},
}

FEATURE_GROUPS = {
    "Connectivity": ["in_degree", "out_degree", "follow_ratio"],
    "Local structure": ["clustering", "triangles", "avg_neighbor_degree"],
    "Relationship": ["reciprocity"],
    "Community": ["community_density", "community_conductance"],
    "Importance": ["pagerank", "betweenness", "k_core"],
    "Temporal": ["creation_similarity", "burst_rate", "interaction_synchrony",
                 "activity_entropy", "growth_rate"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
STRUCTURAL = [f for k, g in FEATURE_GROUPS.items() if k != "Temporal" for f in g]
TEMPORAL = FEATURE_GROUPS["Temporal"]

FEATURE_HELP = {
    "in_degree": "followers (incoming links)",
    "out_degree": "accounts followed (outgoing links)",
    "follow_ratio": "in-degree / out-degree",
    "clustering": "how connected the account's neighbours are to each other",
    "triangles": "closed triads around the account",
    "avg_neighbor_degree": "average degree of its neighbours",
    "reciprocity": "share of its follows that are followed back",
    "community_density": "edge density inside its Louvain community",
    "community_conductance": "external / total links of its community",
    "pagerank": "influence in the network",
    "betweenness": "bridge position in the network",
    "k_core": "k-core number (how deep in the dense core)",
    "creation_similarity": "closeness of creation dates with neighbours (batch creation)",
    "burst_rate": "share of its activity packed into its busiest day",
    "interaction_synchrony": "overlap of active weeks with linked accounts",
    "activity_entropy": "entropy of gaps between events (low = machine-regular)",
    "growth_rate": "new connections per day since creation",
}

# --------------------------------------------------------------------------------------
# 1. Data preparation: synthetic social graph with planted bot clusters
# --------------------------------------------------------------------------------------

COMMON = "today just really love think good time people great new day want see make know way life work back first".split()
TOPICS = [
    "match goal team league coach score win season player fans".split(),
    "album concert song guitar band tour vinyl lyrics playlist stage".split(),
    "python code startup cloud launch bug release data model server".split(),
    "recipe dinner spicy bake coffee brunch flavor kitchen fresh tasty".split(),
    "flight beach hiking trip hotel passport sunset island museum road".split(),
    "election policy budget report minister court vote reform city news".split(),
]
SPAM = "free followers click link win giveaway promo discount subscribe dm cash offer deal now guaranteed".split()
N_COMM = len(TOPICS)


@dataclass
class GraphData:
    params: Params
    G: nx.DiGraph
    y: np.ndarray                 # 1 = bot, 0 = genuine
    group: np.ndarray             # planted group id (-1 = genuine)
    comm: np.ndarray              # true "interest" community of genuine users (bots: -1)
    creation: np.ndarray
    events: list
    texts: list
    feats: pd.DataFrame
    A: np.ndarray                 # undirected adjacency (dense)
    A_dir: np.ndarray             # directed adjacency (dense), A_dir[u, v] = u follows v


def _event_times(rng, k, start, bot, period, noise, scale):
    """Simulate k event timestamps for one account."""
    if k == 0:
        return np.array([])
    if bot:
        gaps = period * np.exp(noise * 1.5 * rng.standard_normal(k))   # clock-like when noise ~ 0
    else:
        gaps = rng.exponential(scale, k)                                # bursty, human-like
    return np.minimum(start + np.cumsum(gaps), 1200.0)


def build_graph(p: Params) -> GraphData:
    rng = np.random.default_rng(p.seed)
    ng, nb = p.n_genuine, p.n_bots
    n = ng + nb
    n_groups = max(1, nb // p.group_size)

    y = np.r_[np.zeros(ng, int), np.ones(nb, int)]
    group = np.r_[-np.ones(ng, int), np.arange(nb) % n_groups]
    comm = np.r_[rng.integers(0, N_COMM, ng), -np.ones(nb, int)]
    pop = np.r_[rng.lognormal(0, 1.0, ng), np.ones(nb)]

    edges = set()

    # ---- genuine users: interest communities, preferential popularity, high reciprocity
    for u in range(ng):
        power = rng.random() < 0.10                       # genuine power users follow a lot
        k = int(np.clip(rng.lognormal(3.3 if power else 2.3, 0.5), 2, 70))
        same = np.where(comm[:ng] == comm[u])[0]
        other = np.where(comm[:ng] != comm[u])[0]
        for _ in range(k):
            pool = same if rng.random() < 0.8 else other
            w = pop[pool]
            v = int(rng.choice(pool, p=w / w.sum()))
            if v != u:
                edges.add((u, v))
                if rng.random() < 0.45:
                    edges.add((v, u))

    # ---- bots
    cam = p.camouflage
    gw = pop[:ng] / pop[:ng].sum()
    for b in range(ng, n):
        g = group[b]
        mates = [x for x in range(ng, n) if group[x] == g and x != b]
        for m in mates:                                   # coordinated group links
            if rng.random() < p.group_density:
                edges.add((b, m))
        stealth = rng.random() < 0.35                     # some bots already look like normal users
        k_out = int(np.clip(rng.lognormal(np.log(30 * (1 - 0.75 * cam) + 8) if not stealth else 2.4, 0.45), 3, 80))
        if cam > 0.4:                                     # mimic a normal community
            c = rng.integers(0, N_COMM)
            pool = np.where(comm[:ng] == c)[0]
            w = pop[pool] / pop[pool].sum()
        else:
            pool, w = np.arange(ng), gw
        for v in rng.choice(pool, size=min(k_out, len(pool)), replace=False, p=w):
            edges.add((b, int(v)))
            if rng.random() < 0.04 + 0.55 * cam + (0.25 if stealth else 0):          # genuine users follow back (attack edges)
                edges.add((int(v), b))

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    out_lists = {u: [] for u in range(n)}
    for u, v in edges:
        out_lists[u].append(v)

    # ---- timestamps (days 0..1200)
    creation = np.empty(n)
    creation[:ng] = rng.uniform(0, 1000, ng)
    cohort = rng.random(ng) < 0.12                    # genuine signup waves (campaigns, launches)
    cdates = rng.uniform(100, 900, 4)
    creation[:ng][cohort] = cdates[rng.integers(0, 4, cohort.sum())] + 4 * rng.standard_normal(cohort.sum())
    creation[:ng] = np.clip(creation[:ng], 0, None)
    centers = rng.uniform(300, 950, n_groups)
    periods = rng.uniform(0.6, 1.6, n_groups)
    for b in range(ng, n):
        creation[b] = max(0.0, centers[group[b]] + p.sync_spread * rng.standard_normal())
    events = []
    for u in range(n):
        outs = out_lists[u]
        rng.shuffle(outs)
        bot = y[u] == 1
        scale = rng.lognormal(np.log(3.0), 0.8)
        start = creation[u] + (rng.exponential(0.5) if bot else rng.exponential(5.0))
        ts = _event_times(rng, len(outs), start, bot,
                          periods[group[u]] if bot else 0, p.timing_noise, scale)
        for v, t in zip(outs, ts):
            G.add_edge(u, v, t=float(t))
        events.append(np.sort(ts))

    # ---- text (only used by the separate content baseline)
    texts = []
    for u in range(n):
        if y[u] == 0:
            words = [rng.choice(TOPICS[comm[u]]) if rng.random() < .5 else rng.choice(COMMON) for _ in range(12)]
        elif p.llm_text:                                   # LLM bot: imitates genuine writing
            t = rng.integers(0, N_COMM)
            words = [rng.choice(TOPICS[t]) if rng.random() < .5 else rng.choice(COMMON) for _ in range(12)]
        else:                                              # classic spam bot
            words = []
            for _ in range(12):
                r = rng.random()
                words.append(rng.choice(SPAM) if r < .4 else rng.choice(COMMON) if r < .7
                             else rng.choice(TOPICS[rng.integers(0, N_COMM)]))
        texts.append(" ".join(words))

    A_dir = nx.to_numpy_array(G, nodelist=range(n), weight=None)
    A = ((A_dir + A_dir.T) > 0).astype(float)
    gd = GraphData(p, G, y, group, comm, creation, events, texts, pd.DataFrame(), A, A_dir)
    gd.feats = compute_features(gd)
    return gd


# --------------------------------------------------------------------------------------
# 2. Feature engineering (Section 6)
# --------------------------------------------------------------------------------------

def compute_features(gd: GraphData) -> pd.DataFrame:
    G = gd.G
    n = G.number_of_nodes()
    U = G.to_undirected()
    f = pd.DataFrame(index=range(n))

    f["in_degree"] = [G.in_degree(i) for i in range(n)]
    f["out_degree"] = [G.out_degree(i) for i in range(n)]
    f["follow_ratio"] = f.in_degree / (f.out_degree + 1)

    f["clustering"] = pd.Series(nx.clustering(U))
    f["triangles"] = pd.Series(nx.triangles(U))
    f["avg_neighbor_degree"] = pd.Series(nx.average_neighbor_degree(U))

    f["reciprocity"] = [len(set(G.successors(i)) & set(G.predecessors(i))) / max(1, G.out_degree(i))
                        for i in range(n)]

    comms = nx.community.louvain_communities(U, seed=gd.params.seed)
    dens, cond = np.zeros(n), np.zeros(n)
    for c in comms:
        c = set(c)
        s = len(c)
        internal = U.subgraph(c).number_of_edges()
        cut = sum(1 for u in c for v in U[u] if v not in c)
        d = internal / (s * (s - 1) / 2) if s > 1 else 0.0
        cd = cut / (2 * internal + cut) if (2 * internal + cut) > 0 else 1.0
        for u in c:
            dens[u], cond[u] = d, cd
    f["community_density"], f["community_conductance"] = dens, cond

    f["pagerank"] = pd.Series(nx.pagerank(G))
    f["betweenness"] = pd.Series(nx.betweenness_centrality(U, k=min(120, n), seed=gd.params.seed))
    U2 = U.copy(); U2.remove_edges_from(nx.selfloop_edges(U2))
    f["k_core"] = pd.Series(nx.core_number(U2))

    # ---- temporal
    cr = gd.creation
    ev = gd.events
    weeks = [set(np.floor(e / 7).astype(int)) for e in ev]
    csim, sync = np.zeros(n), np.zeros(n)
    for i in range(n):
        nbrs = list(U[i])
        if nbrs:
            csim[i] = np.mean(np.exp(-np.abs(cr[nbrs] - cr[i]) / 14.0))
            wi = weeks[i]
            js = [len(wi & weeks[j]) / len(wi | weeks[j]) for j in nbrs if wi or weeks[j]]
            sync[i] = np.mean(js) if js else 0.0
    burst, ent, growth = np.zeros(n), np.zeros(n), np.zeros(n)
    bins = np.linspace(-3, 5, 11)
    for i in range(n):
        e = ev[i]
        if len(e) >= 1:
            days = np.floor(e).astype(int)
            burst[i] = np.bincount(days - days.min()).max() / len(e)
            span = max(e[-1] - cr[i], 1.0)
            growth[i] = U.degree(i) / span
        if len(e) >= 3:
            gaps = np.diff(e)
            h, _ = np.histogram(np.log(np.clip(gaps, 1e-3, None)), bins=bins)
            pr = h[h > 0] / h.sum()
            ent[i] = -(pr * np.log2(pr)).sum()
    f["creation_similarity"], f["burst_rate"], f["interaction_synchrony"] = csim, burst, sync
    f["activity_entropy"], f["growth_rate"] = ent, growth
    return f[FEATURES].astype(float)


def prep(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(X, float), 0, None))


# --------------------------------------------------------------------------------------
# 3. GNNs (dense implementation -> no extra dependencies, fine for demo-size graphs)
# --------------------------------------------------------------------------------------

class SAGE(nn.Module):
    """GraphSAGE (mean aggregator):  h' = ReLU(W [h || mean_neighbours(h)])"""
    def __init__(s, d, h=32):
        super().__init__()
        s.l1, s.l2, s.out = nn.Linear(2 * d, h), nn.Linear(2 * h, h), nn.Linear(h, 1)

    def forward(s, x, A):
        An = A / A.sum(1, keepdim=True).clamp(min=1)
        h = F.relu(s.l1(torch.cat([x, An @ x], 1)))
        h = F.dropout(h, 0.3, s.training)
        h = F.relu(s.l2(torch.cat([h, An @ h], 1)))
        return s.out(h).squeeze(-1)


class GATLayer(nn.Module):
    def __init__(s, din, dout, heads):
        super().__init__()
        s.W = nn.Linear(din, dout * heads, bias=False)
        s.a_src = nn.Parameter(torch.randn(heads, dout) * 0.1)
        s.a_dst = nn.Parameter(torch.randn(heads, dout) * 0.1)
        s.h, s.d = heads, dout

    def forward(s, x, mask):
        n = x.shape[0]
        Wh = s.W(x).view(n, s.h, s.d)
        e = (Wh * s.a_src).sum(-1)[:, None, :] + (Wh * s.a_dst).sum(-1)[None, :, :]   # n,n,h
        e = F.leaky_relu(e, 0.2).masked_fill(~mask[:, :, None], float("-inf"))
        att = F.softmax(e, dim=1)
        att = F.dropout(att, 0.2, s.training)
        return torch.einsum("ijh,jhd->ihd", att, Wh).reshape(n, -1)


class GAT(nn.Module):
    """Graph Attention Network: neighbours are weighted by learned attention."""
    def __init__(s, d, h=8, heads=4):
        super().__init__()
        s.g1, s.g2, s.out = GATLayer(d, h, heads), GATLayer(h * heads, h, heads), nn.Linear(h * heads, 1)

    def forward(s, x, A):
        mask = (A + torch.eye(A.shape[0])) > 0
        h = F.elu(s.g1(x, mask))
        h = F.dropout(h, 0.3, s.training)
        h = F.elu(s.g2(h, mask))
        return s.out(h).squeeze(-1)


def _best_threshold(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0.5, 0.99, 120)))
    best = max(qs, key=lambda t: f1_score(y, s >= t, zero_division=0))
    return float(best)


# --------------------------------------------------------------------------------------
# 4. The detector bank: fit on one graph, apply to any graph (needed for robustness tests)
# --------------------------------------------------------------------------------------

MODEL_ORDER = [
    "Logistic Regression", "Random Forest",
    "Community outlier (unsup.)", "Spectral clustering (unsup.)", "Isolation Forest (unsup.)",
    "GraphSAGE", "GAT",
    "Content baseline (text only)",
]
STRUCT_ONLY = {m: True for m in MODEL_ORDER}
STRUCT_ONLY["Content baseline (text only)"] = False


class DetectorBank:
    def __init__(self, gd: GraphData, split_seed: int = 0, feature_cols=None, models=None, epochs=200):
        self.cols = list(feature_cols or FEATURES)
        self.models_wanted = models or MODEL_ORDER
        self.epochs = epochs
        idx = np.arange(len(gd.y))
        tr, rest = train_test_split(idx, test_size=0.4, stratify=gd.y, random_state=split_seed)
        va, te = train_test_split(rest, test_size=0.5, stratify=gd.y[rest], random_state=split_seed)
        self.tr, self.va, self.te = tr, va, te
        self.fit(gd)

    # -- helpers
    def _X(self, gd):
        return self.scaler.transform(prep(gd.feats[self.cols]))

    def fit(self, gd):
        y, tr, va = gd.y, self.tr, self.va
        self.scaler = StandardScaler().fit(prep(gd.feats[self.cols].iloc[tr]))
        X = self._X(gd)
        self.fitted, self.thr = {}, {}
        want = set(self.models_wanted)

        if "Logistic Regression" in want:
            self.fitted["lr"] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        if "Random Forest" in want:
            self.fitted["rf"] = RandomForestClassifier(300, class_weight="balanced_subsample",
                                                       random_state=0, n_jobs=-1).fit(X[tr], y[tr])
        if "Isolation Forest (unsup.)" in want:
            self.fitted["iso"] = IsolationForest(n_estimators=200, random_state=0).fit(X[tr])
        if "Content baseline (text only)" in want:
            self.tfidf = TfidfVectorizer().fit([gd.texts[i] for i in tr])
            self.fitted["txt"] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                self.tfidf.transform([gd.texts[i] for i in tr]), y[tr])
        for name, cls in (("GraphSAGE", SAGE), ("GAT", GAT)):
            if name in want:
                self.fitted[name] = self._train_gnn(cls, X, gd.A, y)

        # thresholds calibrated on the validation split
        sc = self.score_all(gd)
        for m, s in sc.items():
            self.thr[m] = _best_threshold(y[va], s[va])
        self.base_scores = sc

    def _train_gnn(self, cls, X, A, y):
        torch.manual_seed(0)
        model = cls(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        x, At = torch.tensor(X, dtype=torch.float32), torch.tensor(A, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        pw = torch.tensor((y[self.tr] == 0).sum() / max(1, (y[self.tr] == 1).sum()))
        best, state = -1, None
        for ep in range(self.epochs):
            model.train(); opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x, At)[self.tr], yt[self.tr], pos_weight=pw)
            loss.backward(); opt.step()
            if ep % 5 == 4:
                model.eval()
                with torch.no_grad():
                    s = model(x, At).numpy()
                v = average_precision_score(y[self.va], s[self.va])
                if v > best:
                    best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
        model.load_state_dict(state); model.eval()
        return model

    # -- scoring
    def score_all(self, gd: GraphData) -> dict:
        X = self._X(gd)
        out = {}
        w = set(self.models_wanted)
        if "lr" in self.fitted: out["Logistic Regression"] = self.fitted["lr"].predict_proba(X)[:, 1]
        if "rf" in self.fitted: out["Random Forest"] = self.fitted["rf"].predict_proba(X)[:, 1]
        if "iso" in self.fitted: out["Isolation Forest (unsup.)"] = -self.fitted["iso"].score_samples(X)
        if "Community outlier (unsup.)" in w:
            c = self.cols
            z = lambda name: X[:, c.index(name)] if name in c else 0.0
            out["Community outlier (unsup.)"] = z("community_density") - z("community_conductance") + z("creation_similarity")
        if "Spectral clustering (unsup.)" in w:
            out["Spectral clustering (unsup.)"] = self._spectral(gd)
        if "txt" in self.fitted:
            out["Content baseline (text only)"] = self.fitted["txt"].predict_proba(self.tfidf.transform(gd.texts))[:, 1]
        for name in ("GraphSAGE", "GAT"):
            if name in self.fitted:
                with torch.no_grad():
                    out[name] = torch.sigmoid(self.fitted[name](
                        torch.tensor(X, dtype=torch.float32), torch.tensor(gd.A, dtype=torch.float32))).numpy()
        return {m: out[m] for m in MODEL_ORDER if m in out}

    def _spectral(self, gd):
        n = len(gd.y)
        k = max(6, n // 40)
        try:
            lab = SpectralClustering(k, affinity="precomputed", random_state=0,
                                     assign_labels="kmeans").fit_predict(gd.A + 1e-4)
        except Exception:
            return np.zeros(n)
        s = np.zeros(n)
        for c in range(k):
            m = lab == c
            sz = m.sum()
            if sz > 2:
                s[m] = gd.A[np.ix_(m, m)].sum() / (sz * (sz - 1))
        return s

    def predict(self, scores: dict) -> dict:
        return {m: (s >= self.thr[m]).astype(int) for m, s in scores.items()}


# --------------------------------------------------------------------------------------
# 5. Evaluation (Section 7)
# --------------------------------------------------------------------------------------

def node_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    return dict(
        Precision=precision_score(y, pred, zero_division=0),
        Recall=recall_score(y, pred, zero_division=0),
        F1=f1_score(y, pred, zero_division=0),
        PR_AUC=average_precision_score(y, s),
        ROC_AUC=roc_auc_score(y, s) if len(set(y)) > 1 else float("nan"),
    )


def group_metrics(gd: GraphData, pred: np.ndarray):
    """Cluster-level quality: connected components of flagged accounts vs planted groups."""
    flagged = np.where(pred == 1)[0]
    if len(flagged) == 0:
        return dict(Group_precision=0.0, Group_recall=0.0)
    sub = nx.Graph(gd.A[np.ix_(flagged, flagged)] > 0)
    comps = [[flagged[i] for i in c] for c in nx.connected_components(sub) if len(c) >= 3]
    good = [c for c in comps if gd.y[c].mean() >= 0.5]
    gp = len(good) / len(comps) if comps else 0.0
    gids = [g for g in np.unique(gd.group) if g >= 0]
    hit = sum(1 for g in gids if pred[gd.group == g].mean() >= 0.5)
    return dict(Group_precision=gp, Group_recall=hit / len(gids))


def evaluate(bank: DetectorBank, gd: GraphData, idx=None) -> pd.DataFrame:
    idx = bank.te if idx is None else idx
    sc = bank.score_all(gd)
    rows = []
    for m, s in sc.items():
        met = node_metrics(gd.y[idx], s[idx], bank.thr[m])
        met.update(group_metrics(gd, (s >= bank.thr[m]).astype(int)))
        met["Model"] = m
        met["Structure-only"] = STRUCT_ONLY[m]
        rows.append(met)
    return pd.DataFrame(rows).set_index("Model")


def timing(bank: DetectorBank, gd: GraphData) -> dict:
    """Seconds to predict all test nodes, per model (scores are computed one model at a time)."""
    out = {}
    for m in bank.score_all(gd):
        sub = DetectorBank.__new__(DetectorBank)
        sub.__dict__.update(bank.__dict__)
        sub.models_wanted = [m]
        t0 = time.perf_counter(); sub.score_all(gd); out[m] = time.perf_counter() - t0
    return out


def robustness(bank: DetectorBank, base: Params, scenarios=None):
    """Models are trained once on the baseline graph, then tested on perturbed bot behaviour."""
    scenarios = scenarios or SCENARIOS
    rows, graphs = [], {}
    base_df = None
    for name, change in scenarios.items():
        gd = build_graph(replace(base, **change))
        graphs[name] = gd
        df = evaluate(bank, gd)
        if base_df is None:
            base_df = df
        for m in df.index:
            rows.append(dict(Scenario=name, Model=m, F1=df.loc[m, "F1"],
                             dF1=base_df.loc[m, "F1"] - df.loc[m, "F1"],
                             PR_AUC=df.loc[m, "PR_AUC"]))
    return pd.DataFrame(rows), graphs


def ablation(gd: GraphData, split_seed=0):
    """Feature-group ablation with a Random Forest: drop-one-group and only-one-group."""
    idx = np.arange(len(gd.y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=gd.y, random_state=split_seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=gd.y[rest], random_state=split_seed)

    def run(cols):
        sc = StandardScaler().fit(prep(gd.feats[cols].iloc[tr]))
        X = sc.transform(prep(gd.feats[cols]))
        rf = RandomForestClassifier(200, class_weight="balanced_subsample", random_state=0, n_jobs=-1).fit(X[tr], gd.y[tr])
        s = rf.predict_proba(X)[:, 1]
        thr = _best_threshold(gd.y[va], s[va])
        return f1_score(gd.y[te], s[te] >= thr, zero_division=0), rf

    full, rf = run(FEATURES)
    rows = [dict(Group="ALL FEATURES", Only=full, Without=full)]
    for g, cols in FEATURE_GROUPS.items():
        only, _ = run(cols)
        without, _ = run([c for c in FEATURES if c not in cols])
        rows.append(dict(Group=g, Only=only, Without=without))
    rows.append(dict(Group="Structural only (no temporal)", Only=run(STRUCTURAL)[0], Without=np.nan))
    abl = pd.DataFrame(rows)
    abl["Drop_when_removed"] = full - abl["Without"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return abl, imp


if __name__ == "__main__":
    t = time.time()
    p = Params()
    gd = build_graph(p)
    print("graph", gd.G.number_of_nodes(), gd.G.number_of_edges(), f"{time.time()-t:.1f}s")
    bank = DetectorBank(gd)
    print(f"trained {time.time()-t:.1f}s")
    print(evaluate(bank, gd).round(3).drop(columns=["Structure-only"]))
    r, _ = robustness(bank, p)
    print(r.pivot(index="Model", columns="Scenario", values="F1").round(2).to_string())
    a, imp = ablation(gd)
    print(a.round(3)); print(imp.head(6).round(3))
    print(f"total {time.time()-t:.1f}s")
