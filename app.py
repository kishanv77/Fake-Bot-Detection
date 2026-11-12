"""Streamlit demo:  Fake Account & Bot Network Detection via Graph Structure.
Run:  streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")
from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from botlab.core import (FEATURE_GROUPS, FEATURE_HELP, FEATURES, MODEL_ORDER, SCENARIOS,
                         DetectorBank, Params, ablation, build_graph, evaluate, robustness, timing)

st.set_page_config(page_title="Bot Network Detection", page_icon="🕸️", layout="wide")
BOT, HUM, ERR = "#e4572e", "#2a9d8f", "#f4a300"

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🕸️ Bot Network Lab")
st.sidebar.caption("REVA University · Research Activity 2\nStructure-only fake account detection")
st.sidebar.subheader("1 · Build the synthetic network")
n_g = st.sidebar.slider("Genuine accounts", 200, 800, 450, 50)
n_b = st.sidebar.slider("Bot accounts", 30, 200, 90, 15)
gsz = st.sidebar.slider("Bot group size", 5, 30, 15)
seed = st.sidebar.number_input("Random seed", 0, 999, 7)
params = Params(n_genuine=n_g, n_bots=n_b, group_size=gsz, seed=int(seed))


@st.cache_resource(show_spinner=False)
def get_world(p: Params):
    gd = build_graph(p)
    bank = DetectorBank(gd)
    return gd, bank


@st.cache_resource(show_spinner=False)
def get_layout(p: Params):
    gd, _ = get_world(p)
    U = gd.G.to_undirected()
    return nx.spring_layout(U, seed=1, k=0.35, iterations=60)


@st.cache_resource(show_spinner=False)
def get_timing(p: Params):
    g, b = get_world(p)
    return timing(b, g)


with st.spinner("Generating graph, extracting features and training 8 detectors (≈30 s the first time)…"):
    gd, bank = get_world(params)
    pos = get_layout(params)
scores = bank.score_all(gd)
preds = bank.predict(scores)
met = evaluate(bank, gd)

st.sidebar.success(f"{gd.G.number_of_nodes()} accounts · {gd.G.number_of_edges():,} links · "
                   f"{int(gd.y.sum())} bots in {len(set(gd.group)) - 1} groups")
st.sidebar.caption(f"Split → train {len(bank.tr)} / val {len(bank.va)} / test {len(bank.te)} (stratified). "
                   "Thresholds tuned on validation only.")

st.title("Fake Account & Bot Network Detection via Graph Structure")
st.caption("Working prototype of the research proposal: detect bots using **only who-connects-to-whom and when** "
           "— never the post text.")

tabs = st.tabs(["① Overview", "② Network view", "③ Graph features", "④ Model comparison",
                "⑤ Robustness lab", "⑥ Feature ablation", "⑦ Inspect an account",
                "⑧ Why this matters", "⑨ Text vs structure", "⑩ Future work"])

# ---------------------------------------------------------------- 1. overview
with tabs[0]:
    st.caption("See ⑧–⑩ for motivation, text-vs-structure comparison and roadmap.")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("The problem in one minute")
        st.markdown(
            "Most bot detectors read **what an account says** (posts, profile). Modern AI bots can write "
            "perfectly human text, so those detectors fail. But a bot network still has to *connect* to "
            "other accounts, *get created* and *act* — and that leaves a pattern in the **graph**.\n\n"
            "**This project asks:** how well can we catch fake accounts and coordinated bot networks using "
            "only graph structure + timing, and how does that hold up when bots try to hide?")
        st.markdown("**Research gaps → what you can demo here**")
        st.table(pd.DataFrame({
            "Gap": ["1 Structure-only not isolated", "2 Temporal signals under-studied", "3 Robustness untested"],
            "Where to see it": ["④ Model comparison (text-only baseline kept separate)",
                                "⑥ Feature ablation (structural vs temporal)",
                                "⑤ Robustness lab (camouflage, sparse groups, LLM text…)"]}))
    with c2:
        st.subheader("Methodology (Figure 1 of the proposal)")
        st.graphviz_chart("""
        digraph { rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef3f7", fontname="Helvetica"];
          D [label="Data preparation\\nsynthetic NetworkX graph + planted bot clusters + timestamps"];
          F [label="Feature engineering\\n12 structural + 5 temporal features"];
          M [label="Models\\nLogReg · RandomForest · anomaly detectors · GraphSAGE · GAT"];
          B [label="Content-only baseline\\n(kept separate)", fillcolor="#fde8e1"];
          R [label="Robustness testing\\nperturb bot behaviour"];
          E [label="Evaluation\\nP / R / F1 / PR-AUC / ROC-AUC / ΔF1 / group-level / time"];
          D -> F -> M -> R -> E; D -> B -> E; }
        """)
    st.divider()
    k = st.columns(4)
    best = met.drop(index="Content baseline (text only)")["F1"].idxmax()
    k[0].metric("Accounts / links", f"{gd.G.number_of_nodes()} / {gd.G.number_of_edges():,}")
    k[1].metric("Best structure-only model", best)
    k[2].metric("Its F1 (test)", f"{met.loc[best, 'F1']:.2f}")
    k[3].metric("Text-only baseline F1", f"{met.loc['Content baseline (text only)', 'F1']:.2f}")
    st.info("Data note: the demo uses a **synthetic** network with planted bot groups (the proposal's "
            "'synthetic NetworkX graphs' track). The same pipeline accepts TwiBot-20/22 or Cresci edge lists "
            "once they are converted to `source,target,timestamp` — see README.")

# ---------------------------------------------------------------- 2. network view
with tabs[1]:
    c1, c2 = st.columns([1, 3])
    with c1:
        model = st.selectbox("Detector", MODEL_ORDER[:2] + MODEL_ORDER[2:], index=5)
        mode = st.radio("Colour accounts by", ["Ground truth", "Detector verdict", "Detector mistakes"])
        show_edges = st.checkbox("Draw links", True)
        st.markdown(f"<span style='color:{HUM}'>● genuine</span> &nbsp; <span style='color:{BOT}'>● bot</span> "
                    f"&nbsp; <span style='color:{ERR}'>● mistake</span>", unsafe_allow_html=True)
        row = met.loc[model]
        st.metric("F1 (test)", f"{row.F1:.2f}")
        st.metric("Precision / Recall", f"{row.Precision:.2f} / {row.Recall:.2f}")
        st.metric("Bot groups found / correct", f"{row.Group_recall:.0%} / {row.Group_precision:.0%}")
    with c2:
        p_ = preds[model]
        if mode == "Ground truth":
            col = np.where(gd.y == 1, BOT, HUM)
        elif mode == "Detector verdict":
            col = np.where(p_ == 1, BOT, HUM)
        else:
            col = np.where(p_ != gd.y, ERR, "#c9d1d9")
        fig = go.Figure()
        if show_edges:
            xs, ys = [], []
            for u, v in gd.G.edges():
                xs += [pos[u][0], pos[v][0], None]; ys += [pos[u][1], pos[v][1], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=.3, color="rgba(120,120,120,.25)"),
                                     hoverinfo="skip", showlegend=False))
        xy = np.array([pos[i] for i in range(len(gd.y))])
        txt = [f"Account {i}<br>truth: {'BOT' if gd.y[i] else 'genuine'}<br>score: {scores[model][i]:.2f}"
               f"<br>in/out degree: {int(gd.feats.in_degree[i])}/{int(gd.feats.out_degree[i])}" for i in range(len(gd.y))]
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", text=txt, hoverinfo="text",
                                 marker=dict(size=np.where(gd.y == 1, 8, 6), color=col,
                                             line=dict(width=.4, color="white")), showlegend=False))
        fig.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False,
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bots appear as tight, coordinated clusters (they were created together and follow each other). "
                   "Switch to *Detector verdict* to see what the model flags.")

# ---------------------------------------------------------------- 3. features
with tabs[2]:
    c1, c2 = st.columns([1, 2])
    with c1:
        grp = st.selectbox("Feature group", list(FEATURE_GROUPS))
        feat = st.selectbox("Feature", FEATURE_GROUPS[grp])
        st.markdown(f"**{feat}** — {FEATURE_HELP[feat]}")
        d = gd.feats.assign(label=np.where(gd.y == 1, "bot", "genuine"))
        st.dataframe(d.groupby("label")[feat].describe().round(3))
        st.caption("No single feature proves an account is fake — models combine them.")
    with c2:
        fig = px.histogram(d, x=feat, color="label", barmode="overlay", nbins=40, opacity=.65,
                           color_discrete_map={"bot": BOT, "genuine": HUM}, height=380)
        st.plotly_chart(fig, use_container_width=True)
        sep = {f: abs(gd.feats[f][gd.y == 1].mean() - gd.feats[f][gd.y == 0].mean()) / (gd.feats[f].std() + 1e-9)
               for f in FEATURES}
        s = pd.Series(sep).sort_values()
        fig2 = px.bar(s, orientation="h", height=420, labels={"value": "bot-vs-genuine separation (std units)", "index": ""})
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- 4. models
with tabs[3]:
    st.subheader("All detectors, same stratified split, same test accounts")
    show = met.copy()
    show["Inference (s)"] = pd.Series(get_timing(params))
    show = show.rename(columns={"PR_AUC": "PR-AUC", "ROC_AUC": "ROC-AUC",
                                "Group_precision": "Group prec.", "Group_recall": "Group recall"})
    num = [c for c in show.columns if c != "Structure-only"]
    st.dataframe(show.style.format({c: "{:.3f}" for c in num})
                 .background_gradient(subset=["F1", "PR-AUC"], cmap="Greens"), use_container_width=True)
    m2 = met.reset_index().melt(id_vars="Model", value_vars=["Precision", "Recall", "F1", "PR_AUC"])
    fig = px.bar(m2, x="Model", y="value", color="variable", barmode="group", height=420)
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Why PR-AUC and F1, not accuracy?** Bots are a minority, so a model that says *everyone is genuine* "
                "would still score ~83 % accuracy. Precision/recall/F1/PR-AUC expose that.")

# ---------------------------------------------------------------- 5. robustness
with tabs[4]:
    st.subheader("Do the detectors survive when bots change their behaviour?")
    st.markdown("Models are **trained once on the baseline network**, then tested on networks where the bots behave "
                "differently. ΔF1 = F1(original) − F1(perturbed): smaller is more robust.")

    @st.cache_data(show_spinner=False)
    def run_rob(p: Params):
        _, b = get_world(p)
        return robustness(b, p)[0]

    if st.button("▶ Run all 6 scenarios (~1 min)", type="primary") or st.session_state.get("rob_done") == params:
        with st.spinner("Re-building the network under each scenario…"):
            rob = run_rob(params)
        st.session_state["rob_done"] = params
        piv = rob.pivot(index="Model", columns="Scenario", values="F1").reindex(MODEL_ORDER)[list(SCENARIOS)]
        fig = px.imshow(piv.round(2), text_auto=True, color_continuous_scale="RdYlGn", zmin=0, zmax=1,
                        aspect="auto", height=430, labels=dict(color="F1"))
        st.plotly_chart(fig, use_container_width=True)
        d1 = rob[rob.Scenario != "Baseline (no change)"]
        fig = px.bar(d1, x="Scenario", y="dF1", color="Model", barmode="group", height=400,
                     labels={"dF1": "ΔF1 (higher = less robust)"})
        st.plotly_chart(fig, use_container_width=True)
        st.success("Key read-out: the **text-only baseline collapses only under AI-generated text**, while "
                   "graph methods are unaffected by it — but graph methods are hurt by structural camouflage and "
                   "sparse groups. That is exactly the trade-off the proposal sets out to quantify.")
    else:
        st.caption("Click the button to run the experiment.")

    st.divider()
    st.subheader("🎛️ Live attack: you control the bots")
    a, b_, c, d_, e = st.columns(5)
    cam = a.slider("Camouflage", 0.0, 1.0, 0.0, .05, help="Bots follow normal-looking accounts and get followed back")
    den = b_.slider("Group density", 0.02, 0.6, 0.35, .02, help="How tightly the bots link to each other")
    sp = c.slider("Creation spread (days)", 1, 300, 3, help="Small = accounts created in a batch")
    tn = d_.slider("Timing noise", 0.0, 1.5, 0.15, .05, help="0 = clock-like automation, high = human-like randomness")
    llm = e.checkbox("LLM-written text", False)
    if st.button("Attack!"):
        gd2 = build_graph(replace(params, camouflage=cam, group_density=den, sync_spread=float(sp),
                                  timing_noise=tn, llm_text=llm))
        m2 = evaluate(bank, gd2)
        res = pd.DataFrame({"F1 before": met["F1"], "F1 after": m2["F1"]})
        res["ΔF1"] = res["F1 before"] - res["F1 after"]
        st.dataframe(res.style.format("{:.2f}").background_gradient(subset=["F1 after"], cmap="RdYlGn", vmin=0, vmax=1),
                     use_container_width=True)

# ---------------------------------------------------------------- 6. ablation
with tabs[5]:
    st.subheader("Which signals matter? (Random-Forest feature-group ablation)")

    @st.cache_data(show_spinner=False)
    def run_abl(p: Params):
        g, _ = get_world(p)
        return ablation(g)

    with st.spinner("Re-training with feature groups removed…"):
        abl, imp = run_abl(params)
    c1, c2 = st.columns(2)
    with c1:
        a2 = abl[abl.Group != "ALL FEATURES"].melt(id_vars="Group", value_vars=["Only", "Without"],
                                                    var_name="Setting", value_name="F1")
        a2["Setting"] = a2["Setting"].map({"Only": "Group alone", "Without": "Everything except group"})
        fig = px.bar(a2.dropna(), x="Group", y="F1", color="Setting", barmode="group", height=400)
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        full = abl.loc[abl.Group == "ALL FEATURES", "Only"].iloc[0]
        so = abl.loc[abl.Group.str.startswith("Structural only"), "Only"].iloc[0]
        st.metric("F1 all features", f"{full:.2f}", f"{so - full:+.2f} when temporal features are removed")
    with c2:
        fig = px.bar(imp.sort_values(), orientation="h", height=450, labels={"value": "Random-Forest importance", "index": ""})
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Answers Objective 3/5: do temporal features improve on static structure, and which features are "
               "the most discriminative?")

# ---------------------------------------------------------------- 7. inspect
with tabs[6]:
    c1, c2 = st.columns([1, 2])
    with c1:
        who = st.selectbox("Pick an account", ["a random bot", "a random genuine account", "by number"])
        rs = np.random.default_rng(int(seed))
        if who == "by number":
            node = st.number_input("Account id", 0, len(gd.y) - 1, 0)
        else:
            node = int(rs.choice(np.where(gd.y == (1 if "bot" in who else 0))[0]))
            if st.button("🎲 another"):
                node = int(np.random.choice(np.where(gd.y == (1 if "bot" in who else 0))[0]))
        node = int(node)
        st.markdown(f"### Account {node} — truth: **{'BOT' if gd.y[node] else 'genuine'}**")
        v = pd.DataFrame({"score": {m: scores[m][node] for m in MODEL_ORDER},
                          "threshold": {m: bank.thr[m] for m in MODEL_ORDER}})
        v["verdict"] = np.where(v.score >= v.threshold, "🤖 bot", "✅ genuine")
        st.dataframe(v.round(3), use_container_width=True)
        st.caption(f"Post text (only the content baseline sees this): “{gd.texts[node]}”")
    with c2:
        ego = nx.ego_graph(gd.G.to_undirected(), node, radius=1)
        if ego.number_of_nodes() > 70:
            keep = [node] + list(np.random.default_rng(0).choice([x for x in ego if x != node], 69, replace=False))
            ego = ego.subgraph(keep)
        ep = nx.spring_layout(ego, seed=2)
        fig = go.Figure()
        xs, ys = [], []
        for u, v_ in ego.edges():
            xs += [ep[u][0], ep[v_][0], None]; ys += [ep[u][1], ep[v_][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=.5, color="#aaa"), hoverinfo="skip"))
        nl = list(ego.nodes)
        fig.add_trace(go.Scatter(x=[ep[i][0] for i in nl], y=[ep[i][1] for i in nl], mode="markers+text",
                                 text=[str(i) if i == node else "" for i in nl], textposition="top center",
                                 marker=dict(size=[18 if i == node else 8 for i in nl],
                                             color=[BOT if gd.y[i] else HUM for i in nl],
                                             line=dict(width=1.5, color="black"))))
        fig.update_layout(height=330, showlegend=False, margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False,
                          yaxis_visible=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        med = gd.feats[gd.y == 0].median()
        cmp_ = pd.DataFrame({"this account": gd.feats.loc[node], "genuine median": med})
        cmp_["×  median"] = (cmp_["this account"] / (cmp_["genuine median"] + 1e-9)).round(2)
        st.dataframe(cmp_.round(3), height=330, use_container_width=True)

# ---------------------------------------------------------------- shared helpers for tabs 8-10
TXT = "Content baseline (text only)"
SC_BASE, SC_LLM, SC_CAM = ("Baseline (no change)", "AI-generated text (LLM bots)",
                           "Bots befriend genuine users (camouflage)")
best_so = met.drop(index=TXT)["F1"].idxmax()          # best structure-only model on the baseline test split


def robustness_results(key: str):
    """Tab-5 robustness results (same `run_rob` cache). If not computed yet, offer a button that runs them."""
    if st.session_state.get("rob_done") == params:
        return run_rob(params)
    st.info("These numbers come from the ⑤ Robustness lab, which has not been run for the current settings yet.")
    if st.button("▶ Run all 6 scenarios (~1 min)", key=key, type="primary"):
        with st.spinner("Re-building the network under each scenario…"):
            run_rob(params)
        st.session_state["rob_done"] = params
        st.rerun()
    return None


@st.cache_resource(show_spinner=False)
def get_scenario_world(p: Params, scenario: str):
    """One perturbed network + the frozen detectors' scores on it (used for the example account in ⑨)."""
    _, b = get_world(p)
    g2 = build_graph(replace(p, **SCENARIOS[scenario]))
    return g2, b.score_all(g2)


# ---------------------------------------------------------------- 8. why this matters
with tabs[7]:
    st.subheader("Why this matters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🧩 The problem")
        st.markdown("Most bot detectors judge an account by **what it writes**. Large language models now let bots "
                    "write fluent, human-sounding posts, so text-based detectors lose their main signal. "
                    "Yang & Menczer (2024) document a real botnet whose accounts posted AI-generated content.")
    with c2:
        st.markdown("#### ⚠️ The impact")
        st.markdown("Coordinated fake accounts are used for **scams** (fake giveaways, phishing links), "
                    "**election influence** (making a narrative look popular) and **fake engagement** "
                    "(bought followers, likes and reviews). Undetected, they distort what real users see and trust.")
    with c3:
        st.markdown("#### 🕸️ Our approach")
        st.markdown("Ignore the text entirely. Use only **structure** (who follows whom, how tightly, which "
                    "communities) and **timing** (when accounts were created, how regularly they act). "
                    "A bot can fake its words, but a network still has to be *built* and *operated*.")

    st.divider()
    st.subheader("Before vs after")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("##### 📝 Text-based detection")
        st.markdown("- Input: posts, bio, profile text\n- Catches spam vocabulary and templates\n"
                    "- **Weak spot:** AI-written text looks human\n- Judges each account on its own")
    with b2:
        st.markdown("##### 🕸️ Structure-only detection")
        st.markdown("- Input: follow links + timestamps only\n- Catches dense groups, batch creation, "
                    "clock-like activity\n- **Unaffected by what the text says**\n"
                    "- Weak spot: bots that camouflage their links")
    with b3:
        st.markdown("##### 📏 What this project measures")
        st.markdown("- F1 / PR-AUC of 7 structure-only detectors vs 1 text-only baseline, same test accounts\n"
                    "- Which structural and temporal signals matter (⑥)\n"
                    "- ΔF1 when bots change behaviour, incl. LLM text and camouflage (⑤)")

    st.divider()
    st.subheader("Live evidence from this run")
    rob8 = robustness_results("rob_btn_why")
    if rob8 is not None:
        f1 = rob8.pivot(index="Model", columns="Scenario", values="F1").reindex(MODEL_ORDER)
        so = f1.drop(index=TXT)
        best_llm = so[SC_LLM].idxmax()
        k = st.columns(3)
        k[0].metric("Text-only baseline · LLM-text scenario", f"{f1.loc[TXT, SC_LLM]:.2f}",
                    f"{f1.loc[TXT, SC_LLM] - f1.loc[TXT, SC_BASE]:+.2f} F1 vs baseline network")
        k[1].metric("Best structure-only · LLM-text scenario", f"{so.loc[best_llm, SC_LLM]:.2f}",
                    f"{so.loc[best_llm, SC_LLM] - so.loc[best_llm, SC_BASE]:+.2f} F1 vs baseline network")
        k[2].metric("That structure-only model", best_llm)
        st.caption("F1 on the held-out test accounts. All models were trained once on the baseline network; in the "
                   "LLM-text scenario only the bots' post text changes. Values are read live from the ⑤ results.")
    st.caption("Reference: Yang & Menczer, 2024, “Anatomy of an AI-powered malicious social botnet”.")

# ---------------------------------------------------------------- 9. text vs structure
with tabs[8]:
    st.subheader("Text model vs graph model: same accounts, completely separate inputs")
    st.graphviz_chart(f"""
    digraph {{ rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica"];
      subgraph cluster_t {{ label="Content baseline (text only)"; color="#c9a3d9";
        T1 [label="Post text ONLY\\n(no links, no timestamps)", fillcolor="#f3e8f8"];
        T2 [label="TF-IDF", fillcolor="#f3e8f8"]; T3 [label="Logistic Regression", fillcolor="#f3e8f8"];
        T1 -> T2 -> T3; }}
      subgraph cluster_g {{ label="{best_so} (structure only)"; color="#2a9d8f";
        G1 [label="Follow links + timestamps ONLY\\n(no text)", fillcolor="#e3f4f2"];
        G2 [label="12 structural + 5 temporal features", fillcolor="#e3f4f2"]; G3 [label="{best_so}", fillcolor="#e3f4f2"];
        G1 -> G2 -> G3; }}
      E [label="Same test accounts\\nsame metrics", fillcolor="#eef3f7"];
      T3 -> E; G3 -> E; }}
    """)
    st.caption(f"Structure-only model shown: **{best_so}**, the best structure-only model on the baseline test split "
               "(the one you would deploy). It is kept fixed across scenarios.")

    rob9 = robustness_results("rob_btn_tvs")
    if rob9 is not None:
        lab = {SC_BASE: "(a) Baseline network", SC_LLM: "(b) LLM-written text", SC_CAM: "(c) Camouflage"}
        r = rob9[rob9.Scenario.isin(list(lab)) & rob9.Model.isin([TXT, best_so])].copy()
        r["Scenario"] = r.Scenario.map(lab)
        r["Model"] = r.Model.map({TXT: "Text only: " + TXT, best_so: "Structure only: " + best_so})
        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = px.bar(r, x="Scenario", y="F1", color="Model", barmode="group", text_auto=".2f", height=400,
                         category_orders={"Scenario": list(lab.values())},
                         color_discrete_sequence=["#8d6cab", HUM], range_y=[0, 1.05])
            fig.update_layout(legend=dict(orientation="h", y=-0.2, title=""))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            t = r.pivot(index="Scenario", columns="Model", values=["F1", "PR_AUC"]).reindex(list(lab.values()))
            t.columns = [f"{m.split(':')[0]} · {v.replace('_', '-')}" for v, m in t.columns]
            st.dataframe(t.style.format("{:.2f}"), use_container_width=True)
            f1_9 = rob9.pivot(index="Model", columns="Scenario", values="F1").reindex(MODEL_ORDER).drop(index=TXT)
            st.markdown("**Best structure-only model per scenario** (any of the 7; ties go to the first in ④'s order):")
            st.dataframe(pd.DataFrame({"Model": [f1_9[s].idxmax() for s in lab],
                                       "F1": [f1_9[s].max() for s in lab]}, index=list(lab.values()))
                         .style.format({"F1": "{:.2f}"}), use_container_width=True)
        st.caption("Read it as a trade-off: each family has a different blind spot. All values are read live from "
                   "the ⑤ robustness results (models trained once on the baseline, tested on the same test accounts).")

    st.divider()
    st.subheader("One example account, as each model sees it")
    s1, s2, s3 = st.columns(3)
    kind = s1.radio("Account type", ["Bot", "Genuine"], horizontal=True, key="tvs_kind")
    world = s2.radio("Network", ["Baseline network", "LLM-text scenario"], horizontal=True, key="tvs_world")
    ids = [int(i) for i in sorted(bank.te) if gd.y[i] == (1 if kind == "Bot" else 0)]
    node9 = int(s3.selectbox("Account id (test split)", ids, key=f"tvs_node_{kind}"))
    if world == "Baseline network":
        g9, sc9 = gd, scores
    else:
        with st.spinner("Building the LLM-text network (once)…"):
            g9, sc9 = get_scenario_world(params, SC_LLM)

    def verdict(m):
        s, th = sc9[m][node9], bank.thr[m]
        return f"score {s:.2f} vs threshold {th:.2f} → {'🤖 bot' if s >= th else '✅ genuine'}"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 📝 What the text model sees")
        st.info(f"“{g9.texts[node9]}”")
        st.markdown(f"**{TXT}:** {verdict(TXT)}")
        st.caption("No follower counts, no links, no timestamps: just the words.")
    with c2:
        st.markdown(f"##### 🕸️ What {best_so} sees")
        grp_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
        fv = pd.DataFrame({"group": [grp_of[f] for f in FEATURES],
                           "this account": g9.feats.loc[node9, FEATURES].values,
                           "genuine median": g9.feats[g9.y == 0][FEATURES].median().values}, index=FEATURES)
        st.dataframe(fv.style.format({"this account": "{:.3f}", "genuine median": "{:.3f}"}),
                     height=320, use_container_width=True)
        st.markdown(f"**{best_so}:** {verdict(best_so)}")
        st.caption("No words at all: just numbers computed from links and timing.")
    st.caption(f"Truth: account {node9} is **{'BOT' if g9.y[node9] else 'genuine'}**.")
    if world == "LLM-text scenario":
        same = np.allclose(g9.feats.loc[node9].values, gd.feats.loc[node9].values)
        st.success(f"Compare with *Baseline network*: the post text changed, while this account's graph features are "
                   f"{'**identical**' if same else '**different**'}, so the structure model "
                   f"{'cannot be fooled by the new text' if same else 'also sees a changed network'}.")

# ---------------------------------------------------------------- 10. future work
with tabs[9]:
    st.subheader("Roadmap")
    road = pd.DataFrame([
        ["Structure-only detectors vs separate text baseline", "Isolates what graph structure alone can do",
         "Gap 1", "Done"],
        ["12 structural + 5 temporal features, feature-group ablation",
         "Shows whether timing adds value over static structure", "Gap 2", "Done"],
        ["Robustness scenarios (camouflage, sparse, sync, timing, LLM text)",
         "Measures how detectors fail when bots adapt", "Gap 3", "Done"],
        ["Real datasets: TwiBot-20/22, Cresci", "Synthetic results show the method, not real-world accuracy",
         "Gaps 1-3 (external validity)", "Planned"],
        ["Adversarial training against camouflage", "Camouflage is the scenario that hurts graph models most",
         "Gap 3", "Planned"],
        ["SybilRank and Fraudar baselines", "Classic structure-only defences; cited but not yet compared",
         "Gap 1", "Planned"],
        ["PyTorch Geometric for scale", "Dense-matrix GNNs do not scale to TwiBot-22 size",
         "Gaps 1-3 (scale)", "Planned"],
        ["Streaming / inductive scoring", "Score new accounts as they appear, without retraining",
         "Gap 2 (temporal)", "Planned"],
        ["Explainability (SHAP)", "Per-account reasons for a verdict (today only global RF importances in ⑥ "
         "and feature ratios in ⑦)", "Gap 2 (feature importance)", "Planned"],
    ], columns=["Item", "Why it matters", "Research gap it closes", "Status"])
    k = st.columns(3)
    for col, s in zip(k, ["Done", "In progress", "Planned"]):
        col.metric(s, int((road.Status == s).sum()))
    st.dataframe(road.style.map(lambda v: {"Done": "background-color:#d8f0ec; color:#111",
                                           "In progress": "background-color:#e3ecf7; color:#111",
                                           "Planned": "background-color:#fdf0d5; color:#111"}.get(v, ""),
                                subset=["Status"]),
                 hide_index=True, use_container_width=True)
    st.caption("Only items implemented in this code base are marked Done.")

    st.divider()
    st.subheader("Known limitations")
    st.table(pd.DataFrame({
        "Limitation": ["Synthetic data", "Dense-matrix GNNs", "Ablation with Random Forest only",
                       "Sybil algorithms cited but not implemented"],
        "What it means": ["Planted bot groups and simulated timestamps; numbers illustrate the method, "
                          "not real-world accuracy",
                          "GraphSAGE / GAT use full n×n adjacency matrices; fine at demo size, too large for big graphs",
                          "Feature-group importance is measured for one model; others may rank signals differently",
                          "SybilGuard, SybilRank, CatchSync, CopyCatch and Fraudar appear only as related work"],
        "Addressed by": ["Real datasets (TwiBot-20/22, Cresci)", "PyTorch Geometric",
                         "Repeat ablation per model; SHAP", "SybilRank and Fraudar baselines"],
    }).set_index("Limitation"))
