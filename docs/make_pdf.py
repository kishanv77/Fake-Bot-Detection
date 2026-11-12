from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

OUT = r"C:\Users\Public\Fake Bot Detection\docs\Application_Walkthrough_and_Viva_Guide.pdf"
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14.5, spaceAfter=5)
S = ParagraphStyle("s", parent=B, fontSize=8.8, leading=12, spaceAfter=0)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17, textColor=colors.HexColor("#1b4965"), spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=colors.HexColor("#1b4965"), spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.5, textColor=colors.HexColor("#c4472b"), spaceBefore=6, spaceAfter=2)
Q = ParagraphStyle("q", parent=B, leftIndent=12, backColor=colors.HexColor("#eef5f9"), borderPadding=5, spaceBefore=3, spaceAfter=7)
BUL = ParagraphStyle("bul", parent=B, leftIndent=14, bulletIndent=3, spaceAfter=2)
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=24, textColor=colors.HexColor("#1b4965"), leading=29)

def P(t, st=B): return Paragraph(t, st)
def bl(items): return [Paragraph(i, BUL, bulletText="\u2022") for i in items]
def tbl(rows, widths, head=True):
    data = [[Paragraph(str(c), S) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#b8c4cc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head: st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f1"))]
    t.setStyle(TableStyle(st)); return t

f = []
f += [Spacer(1, 3 * cm), P("Fake Account &amp; Bot Network Detection via Graph Structure", TITLE),
      P("Application walkthrough, backend explained step by step, and a viva guide", H2), Spacer(1, .4 * cm),
      P("REVA University - School of CSE - Research Activity 2 (Major Research Project)<br/>"
        "Center: AI &amp; ML | Guide: Dr. Rashmi C | Project review: 14 November"),
      Spacer(1, .6 * cm),
      P("<b>What this document gives you</b>"),
      *bl(["1. Are all three research gaps covered by the app? (honest answer, with limits)",
           "2. What happens in the backend, in order, from the moment you open the app",
           "3. The ten tabs: what each one is for, what it represents, what runs behind it",
           "4. How to present it to the invigilator: a 2-minute pitch, a 6-minute demo script, likely questions with answers, and what NOT to claim",
           "5. A pre-demo checklist"]),
      Spacer(1, .6 * cm),
      P("<i>All numbers quoted are from the default setting (seed 7, 450 genuine + 90 bots, 6 bot groups). They are from a synthetic network, "
        "so they illustrate the method; they are not real-world accuracy. Numbers shift slightly if you change the seed or sliders.</i>", S),
      PageBreak()]

# ---------------------------------------------------------------- 1
f += [P("1. Are all the research gaps in the application?", H1),
      P("<b>Short answer: yes, each of the three gaps and all five objectives have a working counterpart in the app.</b> "
        "Two things are intentionally simplified (real datasets, and the classic Sybil algorithms); both are listed honestly below so you are never caught off guard."),
      tbl([["Gap in the paper", "Objective", "How the app addresses it", "Where to see it"],
           ["<b>Gap 1</b> Structure-only detection is not isolated; no clean comparison with a content baseline",
            "1, 4",
            "All graph models receive only graph/time features. A separate TF-IDF + Logistic Regression model sees only post text, and is never mixed with the graph models. Same split, same test accounts, same metrics.",
            "Tab 4 (Model comparison): the 'Structure-only' column and the Content baseline row"],
           ["<b>Gap 2</b> Temporal structure and feature importance under-studied",
            "2, 3, 5",
            "17 features: 12 structural + 5 temporal. Feature-group ablation re-trains a Random Forest (a) with only one group, (b) with everything except that group, and (c) structural-only vs structural+temporal. Random-Forest importances rank each feature.",
            "Tab 3 (Graph features) and Tab 6 (Feature ablation)"],
           ["<b>Gap 3</b> Robustness to camouflage and AI-generated content rarely tested",
            "4, 5",
            "Models are trained once on the baseline network, then tested on 5 controlled bot-behaviour changes (camouflage, sparse groups, reduced synchronization, changed timing, LLM-style text). Delta-F1 is reported. Live sliders let you create your own attack.",
            "Tab 5 (Robustness lab)"]],
          [3.4, 1.9, 7.2, 4.5]),
      Spacer(1, .3 * cm),
      P("Objective-by-objective check", H2),
      tbl([["#", "Objective in the paper", "Status in the app"],
           ["1", "Structure-only graph framework using topology + temporal patterns", "Done: build_graph -> compute_features -> DetectorBank"],
           ["2", "Structural features: degree, clustering, reciprocity, community density, centrality", "Done: all of these are computed (plus triangles, avg-neighbour degree, conductance, PageRank, betweenness, k-core)"],
           ["3", "Temporal features: creation synchronization, bursts, cadence", "Done: creation similarity, burst rate, interaction synchrony, activity entropy, growth rate (on synthetic timestamps, as the paper allows)"],
           ["4", "Compare feature classifiers, classical anomaly detectors and GNNs (GraphSAGE, GAT) with a separate content baseline",
            "Done: Logistic Regression, Random Forest; community-outlier, spectral clustering, Isolation Forest; GraphSAGE; GAT; text baseline"],
           ["5", "Evaluate performance and robustness; rank discriminative features", "Done: P/R/F1/PR-AUC/ROC-AUC, delta-F1, group-level precision/recall, inference time, ablation"]],
          [0.8, 8, 7.7]),
      P("Honest limitations (say these before the examiner finds them)", H2),
      *bl(["<b>Data is synthetic.</b> The paper lists TwiBot-20/22 and Cresci as datasets; the demo uses the paper's 'synthetic NetworkX graphs with planted bot clusters and simulated timestamps' track. "
           "The pipeline is dataset-agnostic: real data only needs an edge list (source, target, timestamp) and labels. Loading TwiBot is the next milestone.",
           "<b>Classic Sybil algorithms are not re-implemented.</b> SybilGuard, SybilRank, CatchSync, CopyCatch and Fraudar are in the literature table as related work. "
           "The paper's own methodology names spectral clustering and community-outlier detection as the classical detectors, and those are what is implemented.",
           "<b>Some features are approximated.</b> 'Follow/unfollow burst rate' is measured as the share of an account's events in its busiest day; "
           "'interaction synchrony' is the overlap of active weeks between linked accounts.",
           "<b>The ablation uses a Random Forest only</b>, for speed and clarity.",
           "<b>GNNs are a dense-matrix implementation</b> (no PyTorch Geometric), fine up to ~2,000 nodes; TwiBot-22 scale needs PyTorch Geometric.",
           "<b>Group-level metrics</b> are computed over all accounts; per-account metrics use only the held-out test split."]),
      PageBreak()]

# ---------------------------------------------------------------- 2 backend
f += [P("2. What happens in the backend, step by step", H1),
      P("Everything lives in two files: <b>botlab/core.py</b> (the research pipeline) and <b>app.py</b> (the Streamlit user interface that calls it). "
        "When you open the app, this chain runs once and is then cached, so clicking around is instant."),
      tbl([["Step", "Function", "What it does (one line)"],
           ["0", "Sidebar -> Params", "Your sliders are packed into an immutable settings object; it is the cache key"],
           ["1", "build_graph", "Creates the synthetic social network: genuine users, planted bot groups, timestamps, post text"],
           ["2", "compute_features", "Turns every account into 17 numbers using graph structure and timing"],
           ["3", "DetectorBank.__init__", "Splits accounts 60/20/20, scales features, trains 8 detectors, tunes thresholds"],
           ["4", "score_all / evaluate", "Each detector gives every account a bot-score; metrics are computed on the test accounts"],
           ["5", "robustness", "Rebuilds the network with changed bot behaviour and tests the already-trained models"],
           ["6", "ablation", "Re-trains Random Forests with feature groups removed to find what matters"]],
          [1.2, 4.5, 11.3]),
      P("Step 0 - Parameters and caching", H2),
      P("The sliders (genuine accounts, bots, group size, seed) create a frozen <font face='Courier'>Params</font> object. Streamlit's <font face='Courier'>cache_resource</font> "
        "uses it as the key: same settings = reuse the trained world; change a slider = rebuild. The seed makes the whole experiment repeatable, which matters for a research demo."),
      P("Step 1 - Building the synthetic network (<font face='Courier'>build_graph</font>)", H2),
      P("<b>Genuine users (450).</b> Each belongs to one of 6 interest communities (sports, music, tech, food, travel, news). Each follows about 10 accounts, 80% inside its community, "
        "choosing popular accounts more often (preferential attachment). 45% of follows are followed back (high reciprocity, as in real life). "
        "<b>10% are power users</b> who follow many accounts, and <b>12% were created in signup waves</b> (campaign cohorts) - deliberately included so genuine users sometimes look suspicious and the task is not trivial."),
      P("<b>Bots (90, in 6 coordinated groups of 15).</b> Bots in a group follow each other with probability 0.35 (the 'group density'). "
        "Most bots mass-follow ~30+ popular genuine accounts; <b>35% are 'stealth' bots</b> that follow a normal number of accounts and get followed back more often. "
        "Genuine users rarely follow bots back (4%) in the baseline - those few links are the 'attack edges' that Sybil defences assume are rare."),
      P("<b>Timestamps (days 0-1200).</b> Genuine accounts are created at random times and act in irregular, bursty patterns. Bots in one group are created within about 3 days of each other "
        "(batch creation) and act with clock-like, regular gaps (controlled by 'timing noise'). Each follow link carries a timestamp."),
      P("<b>Post text.</b> Generated only for the separate content baseline. Genuine users write about their community's topic. Classic bots write spam words (free, click, giveaway). "
        "In 'LLM mode' bots write exactly like genuine users, so a text detector cannot tell them apart."),
      P("Step 2 - Feature engineering (<font face='Courier'>compute_features</font>)", H2),
      P("Every account becomes a row of 17 numbers. <b>None uses post text.</b>"),
      tbl([["Group", "Feature", "Computed how", "Why it may reveal bots"],
           ["Connectivity", "in_degree, out_degree, follow_ratio", "Counts of incoming/outgoing links; ratio = in/(out+1)", "Bots follow many, are followed by few"],
           ["Local structure", "clustering, triangles, avg_neighbor_degree", "NetworkX on the undirected graph", "Bot groups form tight triangles"],
           ["Relationship", "reciprocity", "Share of an account's follows that are followed back", "Bots are rarely followed back; groups follow each other"],
           ["Community", "community_density, community_conductance", "Louvain community detection; edges inside / possible edges; external links / total links", "Bot groups form dense, isolated clusters"],
           ["Importance", "pagerank, betweenness, k_core", "Standard NetworkX centralities (betweenness sampled with k=120)", "Position and influence in the network"],
           ["Temporal", "creation_similarity", "Average exp(-|date gap|/14 days) over neighbours", "Neighbours created together = batch creation"],
           ["Temporal", "burst_rate", "Share of an account's events falling on its busiest day", "Automation produces bursts"],
           ["Temporal", "interaction_synchrony", "Average overlap (Jaccard) of active weeks with linked accounts", "Coordinated groups act in the same weeks"],
           ["Temporal", "activity_entropy", "Entropy of the histogram of log time-gaps between events", "Low entropy = machine-like regular timing"],
           ["Temporal", "growth_rate", "Connections per day since creation", "Unnaturally fast network growth"]],
          [2.2, 4.2, 6.3, 4.3]),
      P("Step 3 - Split, preprocessing and training (<font face='Courier'>DetectorBank</font>)", H2),
      *bl(["<b>Stratified split 60 / 20 / 20</b> (train / validation / test) so the bot fraction is the same in each. Same split for every model (as the paper requires).",
           "<b>Preprocessing:</b> log(1+x) to tame heavy-tailed counts, then standardization fitted on the training accounts only (no leakage).",
           "<b>Threshold tuning:</b> a model outputs a score; the cut-off for 'bot' is chosen to maximize F1 on the <i>validation</i> set, never on the test set."]),
      tbl([["Model", "Type", "How it decides"],
           ["Logistic Regression", "Supervised, feature-based", "Weighted sum of features (class-balanced)"],
           ["Random Forest", "Supervised, feature-based", "300 decision trees voting; gives feature importances"],
           ["Community outlier", "Unsupervised anomaly", "Score = community density - conductance + creation similarity (z-scored); no labels used to build the score"],
           ["Spectral clustering", "Unsupervised anomaly", "Clusters the adjacency matrix; accounts in unusually dense clusters score high"],
           ["Isolation Forest", "Unsupervised anomaly", "Flags accounts whose feature vectors are easy to isolate (statistical outliers)"],
           ["GraphSAGE", "Graph Neural Network", "Each account's vector is combined with the <i>average of its neighbours'</i> vectors, two layers deep"],
           ["GAT", "Graph Neural Network", "Like GraphSAGE but learns <i>attention weights</i>: which neighbours matter more (4 heads)"],
           ["Content baseline", "Text only (separate)", "TF-IDF of post text + Logistic Regression; never sees the graph"]],
          [3.3, 3.7, 10]),
      P("The two GNNs train for 200 epochs with a class-weighted loss and keep the epoch with the best validation PR-AUC. They receive the 17 graph/time features per account "
        "<i>plus</i> message passing over the follow graph - that is what makes them 'graph' models."),
      P("Step 4 - Evaluation (<font face='Courier'>evaluate</font>, <font face='Courier'>timing</font>)", H2),
      *bl(["<b>Precision</b> = of accounts flagged, how many are really bots. <b>Recall</b> = of real bots, how many were caught. <b>F1</b> balances them. "
           "<b>PR-AUC / ROC-AUC</b> measure ranking quality across all thresholds. Accuracy is deliberately not reported (bots are only ~17%, so 'say everyone is genuine' scores 83%).",
           "<b>Group-level detection:</b> flagged accounts that are connected form clusters. Group precision = share of flagged clusters (3+ accounts) that are mostly bots. "
           "Group recall = share of planted bot groups where at least half the members were flagged.",
           "<b>Inference time:</b> seconds to score all accounts, per model."]),
      P("Step 5 - Robustness testing (<font face='Courier'>robustness</font>)", H2),
      P("This is the paper's central experiment. The trained models are <b>frozen</b>. For each scenario the network is rebuilt with the <b>same seed</b> (same genuine users) but changed bot behaviour, and the frozen models are tested on the <b>same test accounts</b>. "
        "Delta-F1 = F1(original) - F1(perturbed). Smaller drop = more robust."),
      tbl([["Scenario", "What changes in build_graph", "Real-world meaning"],
           ["Camouflage", "camouflage = 0.85: bots follow a normal number of accounts in one community; genuine users follow back ~50%", "Bots befriend genuine users to look normal"],
           ["Sparse groups", "group_density 0.35 -> 0.06", "Bots coordinate loosely, not as a dense block"],
           ["Reduced synchronization", "creation spread 3 -> 250 days", "Bot accounts are created slowly over time"],
           ["Changed timing", "timing_noise 0.15 -> 1.3", "Bots randomize their activity to look human"],
           ["AI-generated text", "bots write like genuine users", "LLM bots; only the text baseline is expected to change"]],
          [3.3, 8, 5.7]),
      P("Observed with the default seed:", H3),
      tbl([["Model", "Baseline F1", "Camouflage", "Sparse groups", "Less sync", "New timing", "AI text"],
           ["Logistic Regression", "1.00", "0.97", "0.88", "0.91", "0.97", "1.00"],
           ["Random Forest", "1.00", "1.00", "0.88", "1.00", "0.94", "1.00"],
           ["GraphSAGE", "1.00", "<b>0.20</b>", "0.88", "0.91", "0.88", "1.00"],
           ["GAT", "0.94", "0.84", "<b>0.00</b>", "<b>0.00</b>", "<b>0.00</b>", "0.94"],
           ["Community outlier (unsup.)", "0.89", "0.29", "0.11", "<b>0.00</b>", "0.89", "0.89"],
           ["Text-only baseline", "0.97", "0.94", "1.00", "0.97", "0.97", "<b>0.00</b>"]],
          [4.4, 2.0, 2.1, 2.3, 2.0, 2.1, 2.1]),
      P("<b>The story:</b> graph methods are completely immune to AI-written text, but are hurt by structural camouflage, sparse groups and desynchronization - "
        "and the text baseline is the mirror image. Neither family is safe everywhere, which is exactly the trade-off the research question asks about. "
        "The GAT collapses (F1 0.00) under several changes: it learned thresholds tied to the baseline feature distribution, which is itself a finding about GNN brittleness when behaviour drifts."),
      P("Step 6 - Feature ablation (<font face='Courier'>ablation</font>)", H2),
      P("A Random Forest is re-trained several times: with <b>only</b> one feature group, and with <b>everything except</b> that group, plus structural-only vs all features. "
        "Observed: removing the temporal group drops F1 from 1.00 to 0.91, and the temporal group alone reaches 1.00, so temporal features add real value over static structure. "
        "Top-ranked individual features: activity entropy, reciprocity, creation-time similarity, follow ratio, interaction synchrony."),
      PageBreak()]

# ---------------------------------------------------------------- 3 tabs
f += [P("3. The ten tabs: purpose, meaning, and what runs behind them", H1),
      P("Tabs 1-7 are the research pipeline. Tabs 8-10 are presentation tabs added for the project review: motivation, "
        "a direct text-vs-structure comparison, and an honest roadmap. They add no new models and change no metrics; "
        "every number they show is read live from results the app has already computed.")]
tabs = [
    ("Tab 1 - Overview", "Sets the story before showing numbers.",
     "The problem in plain words, the methodology diagram (Figure 1 of the paper), and a table that maps each research gap to the tab that answers it. Four headline numbers: network size, best structure-only model, its F1, and the text baseline F1.",
     "No new computation; it reads the already-trained models.",
     "Start here. Say: 'Existing detectors read text; AI bots beat them; we test whether structure and timing alone can detect bots, and how robust that is.'"),
    ("Tab 2 - Network view", "Shows that bot networks are visible in the graph.",
     "The whole network drawn with a force-directed layout. Dots are accounts, lines are follows. Colour by ground truth (green genuine, red bot), by the chosen detector's verdict, or by its mistakes (orange). Side panel shows F1, precision/recall and group detection for the chosen model.",
     "Spring layout of the undirected graph (cached), predictions from score_all thresholded at the validation-tuned cut-off.",
     "Switch between 'Ground truth' and 'Detector verdict' and show they match. Pick a weak model (Isolation Forest) to show mistakes. Hover a dot to see its score and degrees."),
    ("Tab 3 - Graph features", "Shows WHY detection works: the signals themselves.",
     "Pick a feature group and feature, read its definition, and see overlaid histograms for bots vs genuine accounts plus a bar chart of how strongly each feature separates the two classes.",
     "Reads the 17-column feature table; separation = |mean_bot - mean_genuine| / std.",
     "Show a clear one (activity_entropy or follow_ratio), then a weak one, and repeat the paper's caution: 'no single feature is proof of a bot; that is why models combine them.'"),
    ("Tab 4 - Model comparison", "Answers Gap 1: structure-only vs content-based, fairly.",
     "A table of Precision, Recall, F1, PR-AUC, ROC-AUC, group precision/recall, a Structure-only flag, and inference time for all 8 models, plus a bar chart.",
     "evaluate on the held-out test accounts and timing.",
     "Point to the Structure-only column: seven graph models vs one text model, same split. Explain why accuracy is not shown (imbalanced classes)."),
    ("Tab 5 - Robustness lab", "Answers Gap 3: what happens when bots adapt.",
     "Button 'Run all 6 scenarios' produces an F1 heatmap (models x scenarios) and a bar chart of delta-F1. Below, the 'Live attack' sliders (camouflage, group density, creation spread, timing noise, LLM text) let you design your own bot behaviour and see before/after F1 for every model.",
     "robustness: rebuilds the network per scenario, tests frozen models. Live attack: build_graph with your slider values, then evaluate.",
     "The showpiece. Click Run (cache it before the review), read the heatmap row for the text baseline (dies at AI text) and the GraphSAGE row (dies at camouflage). Then drag camouflage up live."),
    ("Tab 6 - Feature ablation", "Answers Gap 2: which signals matter, and do temporal ones help?",
     "Bar chart of F1 with a feature group alone vs everything except that group; a metric showing the effect of removing temporal features; and a ranked list of individual feature importances.",
     "ablation: repeated Random Forest training on feature subsets.",
     "Say: 'Temporal features add measurable value over static structure (F1 1.00 vs 0.91), which fills the gap that Sybil defences use only static graphs.'"),
    ("Tab 7 - Inspect an account", "Makes the result concrete and explainable for one account.",
     "Choose a random bot or genuine account (or by id). See each model's score, threshold and verdict; its one-hop neighbourhood drawn as a mini graph; and its features compared with the median genuine account (ratio column).",
     "Looks up the score of that node from score_all and its row in the feature table.",
     "Great for questions like 'why was this flagged?': show the ratio column - e.g. follows 2x more accounts, reciprocity near zero. Also shows the post text that only the content baseline sees."),
    ("Tab 8 - Why this matters", "Motivates the project in plain language before any technical detail.",
     "Three short panels: the problem (AI-written bots defeat text-based detectors), the impact (scams, election influence, fake engagement) and our approach (structure + timing only). "
     "A three-column 'before vs after' explainer: text-based detection, structure-only detection, and what this project measures. "
     "A live callout with two numbers: the text-only baseline F1 under the LLM-text scenario, and the best structure-only F1 under the same scenario, each with its change from the baseline network. "
     "The reference cited is Yang &amp; Menczer, 2024, 'Anatomy of an AI-powered malicious social botnet'.",
     "No new computation. The callout reads the cached Tab 5 robustness results (same run_rob cache). If they are not computed yet, the tab shows a button that runs them once and then every tab reuses them.",
     "Use it as the opening slide. Read the callout aloud: the text model's F1 falls to the number on screen while the graph model's stays put. Say 'this is why we study structure'."),
    ("Tab 9 - Text vs structure", "Shows side by side that the two families use completely separate inputs and fail in different situations.",
     "A diagram with two separate lanes: post text -> TF-IDF -> Logistic Regression for the content baseline, and links + timestamps -> 17 features -> the best structure-only model. "
     "A grouped bar chart and table comparing the two models' F1 and PR-AUC on (a) the baseline network, (b) the LLM-text scenario and (c) the camouflage scenario, plus the best structure-only model per scenario. "
     "Below, one example test account: its post text with the text model's score and verdict, and its 17 graph features (vs the genuine median) with the graph model's score and verdict. "
     "A 'Network' switch moves the same account into the LLM-text scenario.",
     "Reuses the existing Content baseline (text only); no new detector is built. Comparison numbers come from the cached Tab 5 results. The LLM-text network for the example account is built once with build_graph and cached; the frozen models score it.",
     "Point at the two lanes: 'the text model never sees the graph, the graph model never sees the text'. Then pick a bot, switch Network to 'LLM-text scenario': the post now reads like a genuine user and the text verdict flips, while the graph features are identical (the app confirms this) and the graph verdict does not change. Finish on the camouflage bars: the graph family has its own blind spot."),
    ("Tab 10 - Future work", "Shows the examiner you know the limits and have a concrete plan.",
     "A roadmap table: item, why it matters, which research gap it closes, and status (Done / In progress / Planned). Done: structure-only vs text baseline, structural + temporal features with ablation, robustness scenarios. "
     "Planned: real datasets (TwiBot-20/22, Cresci), adversarial training against camouflage, SybilRank and Fraudar baselines, PyTorch Geometric for scale, streaming / inductive scoring, explainability (SHAP). "
     "A 'Known limitations' table: synthetic data, dense-matrix GNNs, ablation with Random Forest only, Sybil algorithms cited but not implemented, each mapped to the roadmap item that addresses it.",
     "Static content; no computation.",
     "Close the demo here. Say the limitations yourself before you are asked, then point to the roadmap row that fixes each one. Only the three implemented items are marked Done; do not claim more."),
]
for name, purpose, rep, back, tip in tabs:
    f.append(KeepTogether([P(name, H2),
        tbl([["Purpose", purpose], ["What you see", rep], ["Backend behind it", back], ["How to present it", tip]], [3, 14], head=False)]))
f += [Spacer(1, .3 * cm),
      P("Sidebar: the sliders rebuild the whole experiment (new graph, new features, new training). The green box shows the network size, and the line below it confirms the 324 / 108 / 108 train / validation / test split.", B),
      PageBreak()]

# ---------------------------------------------------------------- 4 viva
f += [P("4. How to explain this to the invigilator", H1),
      P("The 2-minute pitch (memorize the shape, not the words)", H2),
      P("&ldquo;Fake accounts and bot networks are a growing problem, and most detectors rely on what an account <i>posts</i>. With AI language models, bots can now write convincing posts, "
        "so text-based detection is becoming unreliable. Our project asks a different question: <b>can we detect bots using only graph structure and timing</b> - who follows whom, when accounts were created, how regularly they act - "
        "and how well does that hold up when bots adapt?<br/><br/>"
        "We found three gaps in the literature: structure-only detection is never isolated from text; temporal signals are under-studied; and robustness to camouflage and AI content is rarely tested. "
        "We built a working framework that generates a social network with hidden bot groups, extracts 17 structural and temporal features, and compares eight detectors - from logistic regression to graph neural networks - "
        "against a separate text-only baseline. We then stress-test all of them under five bot-behaviour changes and measure the drop in F1.<br/><br/>"
        "Our key finding so far: graph methods are unaffected by AI-written text, which breaks the text baseline completely, but graph methods are vulnerable to camouflage and sparse groups. "
        "So the two approaches fail in opposite ways, which motivates structure-based defences.&rdquo;", Q),
      P("The 6-minute live demo script", H2),
      tbl([["Min", "Do this", "Say this"],
           ["0:00", "Tab 8", "Why this matters: AI-written bots beat text detectors. Read the live callout: text F1 under LLM text vs best structure-only F1."],
           ["0:35", "Tab 1", "Three gaps, and which tab answers which. Point at the methodology diagram."],
           ["1:05", "Tab 2, 'Ground truth' then 'Detector verdict'", "Red dots are bots in 6 groups, created together and following each other. The graph model finds them without reading any text."],
           ["1:45", "Tab 3", "Why: show activity_entropy or follow_ratio histograms. Bots are machine-regular and mass-follow."],
           ["2:15", "Tab 4", "Eight models, same split. Structure-only column. F1 and PR-AUC, not accuracy, because bots are a minority."],
           ["2:45", "Tab 9", "Two separate lanes. Pick a bot, switch to the LLM-text network: text verdict flips, graph features identical, graph verdict holds."],
           ["3:30", "Tab 5 heatmap (pre-run), then live attack", "Text baseline fails only on AI text; graph models fail on camouflage. Raise camouflage live and watch them degrade."],
           ["4:30", "Tab 6", "Temporal features add value over static structure; here are the top features."],
           ["5:00", "Tab 7", "Any single account: scores, neighbourhood, and how its features compare with a typical genuine user."],
           ["5:30", "Tab 10", "Known limitations in our own words, and the roadmap row that fixes each one. Close."]],
          [1.3, 5.2, 10.5]),
      P("Questions the invigilator may ask, with answers", H2)]
qa = [
    ("Is this real data?", "No - the demo uses the synthetic-graph track in our methodology: NetworkX with planted bot clusters and simulated timestamps. That gives ground truth for group-level evaluation and lets us control bot behaviour for robustness tests. The pipeline is dataset-agnostic, and TwiBot-20/22 and Cresci are the next step."),
    ("Your F1 is 1.0 - isn't that suspicious?", "On a synthetic baseline, yes it is easy, so we added stealth bots, genuine power users and genuine signup waves, and the unsupervised detectors score 0.68-0.89. The meaningful result is not the baseline score but the drop under perturbation, where F1 falls to 0.00-0.20 for some models. Real data will be harder; we will not claim these numbers transfer."),
    ("Why not just use accuracy?", "Only about 17% of accounts are bots, so a model that says 'everyone is genuine' gets 83% accuracy. Precision, recall, F1 and PR-AUC expose that."),
    ("What exactly is 'structure-only'?", "Models see node-level numbers computed from links and timestamps - degrees, clustering, reciprocity, community density, creation-time similarity - and, for the GNNs, neighbours' numbers. They never see post text. The text baseline is trained separately and never sees the graph."),
    ("Why do GNNs help?", "They combine an account's own features with those of its neighbours, so they can learn 'accounts surrounded by bot-like accounts are suspicious'. GraphSAGE averages neighbours; GAT learns attention weights over neighbours."),
    ("Why did the GNNs fail under some scenarios?", "They were trained on the baseline bot behaviour and learned boundaries tied to it. When the bots shift (for example timing becomes random), the feature distributions move outside what the model learned. That brittleness under distribution shift is a finding we will study further, for example with training on mixed behaviours."),
    ("How do you prevent data leakage?", "Stratified split fixed once; scaler and models fit on training accounts only; thresholds tuned on validation; metrics on test accounts only; robustness tests re-use the same test accounts."),
    ("What is Louvain?", "A community-detection algorithm that groups nodes to maximize modularity. We use it to find the community each account belongs to and compute that community's density and conductance."),
    ("How are coordinated groups detected, not just single accounts?", "We take the flagged accounts, connect those with links, and treat each connected cluster of 3+ accounts as a detected group; group precision and recall compare them with the planted groups."),
    ("What are the limitations?", "Synthetic data; no re-implementation of SybilRank/CatchSync/Fraudar (only cited); ablation done with one model; dense-matrix GNNs. We list them in our report and will address them with real datasets and PyTorch Geometric."),
    ("What is your contribution beyond existing work?", "A controlled framework that separates structure-only detection from content, adds temporal analysis, and quantifies robustness of classical and GNN detectors under specific bot-behaviour changes in one reproducible pipeline."),
    ("What will you do next?", "Load TwiBot-20/22 and Cresci, scale GNNs with PyTorch Geometric, add adversarial training against camouflage, and include the classic Sybil baselines for a complete comparison. Tab 10 shows the full roadmap with the research gap each item closes."),
]
for q, a in qa:
    f.append(KeepTogether([P(f"<b>Q: {q}</b>", B), P(a, Q)]))
f += [P("What NOT to say", H2),
      *bl(["Do not say the app detects bots on real Twitter/X data - it does not yet.",
           "Do not say GNNs are 'the best' - on this data Random Forest and GraphSAGE tie at baseline, and GAT is the most brittle.",
           "Do not present 1.00 as an achievement; present the robustness drop as the result.",
           "Do not claim the Sybil algorithms are implemented."]),
      P("5. Pre-demo checklist", H1),
      *bl(["Run the app 30 minutes before: double-click run_demo.bat, open http://localhost:8501, wait for the green box to appear in the sidebar.",
           "Keep the default seed (7) so the numbers in this PDF match the screen.",
           "Open Tab 5 and press 'Run all 6 scenarios' once so the heatmap is cached (about 1 minute). Tabs 8 and 9 reuse these results, so their numbers appear immediately afterwards. Open Tab 6 once as well.",
           "In Tab 9, switch the Network to 'LLM-text scenario' once before the review so that network is cached, then switch back.",
           "Do not change the sidebar sliders during the demo unless you want to re-train (about 30 seconds).",
           "Have a backup: take screenshots of tabs 2, 4, 5, 6 in case the laptop misbehaves.",
           "Know where the code is: botlab/core.py (pipeline) and app.py (interface)."])]


def foot(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "Fake Account & Bot Network Detection - Application walkthrough and viva guide")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}"); c.restoreState()

SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                  title="Application Walkthrough and Viva Guide").build(f, onFirstPage=foot, onLaterPages=foot)
print("ok")
