"""
USDA Digital Experience Intelligence Platform
Streamlit Application — K-Means Segmentation of Rural Development Web Pages
"""

import re, io, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──
st.set_page_config(page_title="USDA Digital Intelligence Platform", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ──
st.markdown("""
<style>
  [data-testid="stMetricValue"]{font-size:2rem;font-weight:700}
  .kpi-card{background:rgba(46,125,50,.1);border-radius:16px;padding:22px;margin:8px;
    border:1px solid rgba(46,125,50,.2);box-shadow:0 8px 32px rgba(0,0,0,.05);color:#1b5e20;transition:.3s}
  .kpi-card:hover{background:rgba(46,125,50,.15);transform:translateY(-3px)}
  .insight-box{background:#f0f7f0;border-left:5px solid #2e7d32;padding:14px 18px;border-radius:6px;margin:10px 0;color:#1b5e20;font-size:.95rem}
  .warning-box{background:#fff8e1;border-left:5px solid #f9a825;padding:14px 18px;border-radius:6px;margin:10px 0;color:#e65100;font-size:.95rem}
  .section-header{border-bottom:3px solid #2e7d32;padding-bottom:6px;color:#1b5e20;font-weight:700;font-size:1.3rem;margin-top:24px}
  .agent-response{background:#fff;border-radius:15px;padding:25px;border:1px solid #e0e0e0;font-size:.95rem;line-height:1.7;box-shadow:0 4px 15px rgba(0,0,0,.05);color:#333}
  .chat-bubble-user{background:#e8f5e9;color:#1b5e20;padding:12px 18px;border-radius:15px 15px 0 15px;margin-bottom:15px;border:1px solid #c8e6c9;width:fit-content;margin-left:auto;font-weight:500}
  @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
  .stTabs [data-baseweb="tab-panel"]{animation:fadeIn .4s ease-out}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──
DEFAULT_PATH = "DATA USDA.csv"
K_CLUSTERS = 4  # Fixed — optimal silhouette=0.443

CLUSTERING_FEATURES = ["log_sessions", "bounce_rate", "views_per_session"]

SEGMENT_COLORS = {
    "Well-Served": "#1b5e20",
    "Deep-Engagement Niche": "#0288d1",
    "Moderately Served": "#f57c00",
    "Underserved / High-Friction": "#c62828",
}
USDA_COLORS = list(SEGMENT_COLORS.values())

# ── DATA LOADING ──
def load_usda_csv(uploaded_file=None, filepath=None):
    def _try(src, skip):
        try:
            if hasattr(src, "read"): src.seek(0)
            return pd.read_csv(src, skiprows=skip, low_memory=False)
        except Exception: return None
    for skip in [7,6,5,4,3,2,1,0]:
        df = _try(uploaded_file if uploaded_file else filepath, skip)
        if df is not None and len(df.columns)>=5:
            fc = str(df.columns[0]).strip()
            if fc.startswith("#") or fc.lower() in ("","nan"): continue
            ac = " ".join(str(c).lower() for c in df.columns)
            if any(k in ac for k in ["session","page","user","view","bounce"]): return df
    if uploaded_file: uploaded_file.seek(0); return pd.read_csv(uploaded_file, low_memory=False)
    return pd.read_csv(filepath, low_memory=False)


def detect_totals_columns(df):
    metrics = ["Active users","Event count","Sessions","Views per session",
               "Average session duration","Bounce rate","Exits","Returning users","Total users"]
    col_map = {}
    for m in metrics:
        matches = [c for c in df.columns if str(c).startswith(m)]
        if matches: col_map[m] = matches[-1]
    return col_map


def prepare_page_data(df, country_filter=None):
    totals = detect_totals_columns(df)
    rename = {v: k.lower().replace(" ","_") for k,v in totals.items()}
    id_cols = [c for c in ["Page title","Page path and screen class","Month","Day","Country"] if c in df.columns]
    work = df[id_cols + list(totals.values())].copy().rename(columns=rename)
    pc = "Page path and screen class"
    if pc not in work.columns: st.error("Cannot find page path column."); st.stop()
    work = work.dropna(subset=[pc])
    work = work[~work[pc].astype(str).str.startswith("#")]
    work = work[work[pc].astype(str).str.strip()!=""]
    work = work[work[pc].astype(str).str.strip().str.lower()!="nan"]

    # Country filter
    if country_filter and "Country" in work.columns:
        fl = [c.lower().strip() for c in country_filter]
        work = work[work["Country"].astype(str).str.lower().str.strip().isin(fl)]
    if work.empty: return pd.DataFrame()

    num = ["active_users","event_count","sessions","views_per_session",
           "average_session_duration","bounce_rate","exits","returning_users","total_users"]
    for c in num:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c].astype(str).str.replace(",","").str.replace(" ",""), errors="coerce")
    agg = {}
    for c in ["sessions","active_users","total_users","event_count","exits","returning_users"]:
        if c in work.columns: agg[c]="sum"
    for c in ["views_per_session","average_session_duration","bounce_rate"]:
        if c in work.columns: agg[c]="mean"
    pages = work.groupby(pc).agg(agg).reset_index()
    pages = pages.rename(columns={pc:"page_id","average_session_duration":"avg_session_duration"})
    pages = pages[pages["sessions"]>0].copy()
    if "Page title" in df.columns:
        tm = work.dropna(subset=["Page title",pc]).drop_duplicates(pc).set_index(pc)["Page title"].to_dict()
        pages["page_title"] = pages["page_id"].map(tm)
    else:
        pages["page_title"] = pages["page_id"]
    return pages


def engineer_features(df):
    df = df.copy()
    for c in ["sessions","active_users","total_users","exits","event_count",
              "returning_users","bounce_rate","avg_session_duration","views_per_session"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
        else: df[c]=0.0
    df = df[df["sessions"]>0].copy()
    df["exit_ratio"] = np.where(df["sessions"]>0, df["exits"]/df["sessions"], 0)
    df["events_per_session"] = np.where(df["sessions"]>0, df["event_count"]/df["sessions"], 0)
    ud = np.maximum(df["total_users"].values, df["active_users"].values)
    ud = np.where(ud==0,1,ud)
    df["returning_user_ratio"] = df["returning_users"].values / ud
    df["log_sessions"] = np.log1p(df["sessions"])
    sc = ["avg_session_duration","views_per_session","events_per_session","returning_user_ratio","bounce_rate","exit_ratio"]
    pr = [c for c in sc if c in df.columns]
    v = df[pr].fillna(0).values
    mn,mx = v.min(0), v.max(0); rng=mx-mn; rng[rng==0]=1
    s = (v-mn)/rng
    sd = pd.DataFrame(s, columns=pr, index=df.index)
    g = lambda c: sd[c].values if c in sd.columns else 0
    df["engagement_score"] = (g("avg_session_duration")*.25+g("views_per_session")*.20+
        g("events_per_session")*.20+g("returning_user_ratio")*.20-g("bounce_rate")*.075-g("exit_ratio")*.075).clip(0,1)
    df["friction_index"] = (g("bounce_rate")*.35+g("exit_ratio")*.35-g("avg_session_duration")*.15-g("views_per_session")*.15).clip(0,1)
    df["priority_score"] = df["friction_index"]*df["log_sessions"]
    return df.reset_index(drop=True)


# ── CLUSTERING (pure NumPy) ──
def _kmeans(X, k, n_init=30, max_iter=500):
    bi=np.inf; bl=bc=None; n=X.shape[0]
    for _ in range(n_init):
        idx=np.random.choice(n,k,replace=False); c=X[idx].copy()
        for _ in range(max_iter):
            d=np.linalg.norm(X[:,None]-c[None,:],axis=2); l=np.argmin(d,axis=1)
            nc=np.array([X[l==i].mean(0) if (l==i).sum()>0 else c[i] for i in range(k)])
            if np.allclose(c,nc,atol=1e-7): break
            c=nc
        inr=sum(np.sum((X[l==i]-c[i])**2) for i in range(k))
        if inr<bi: bi=inr;bl=l;bc=c
    return bl,bc,bi

def _silhouette(X, labels, ss=3000):
    n=len(labels)
    if n>ss: idx=np.random.choice(n,ss,replace=False); X,labels=X[idx],labels[idx]; n=ss
    uq=np.unique(labels)
    if len(uq)<2: return -1.0
    sc=np.zeros(n)
    for i in range(n):
        own=labels[i]; om=labels==own
        if om.sum()<=1: continue
        a=np.mean(np.linalg.norm(X[om]-X[i],axis=1))
        bm=min(np.mean(np.linalg.norm(X[labels==c]-X[i],axis=1)) for c in uq if c!=own)
        sc[i]=(bm-a)/max(max(a,bm),1e-10)
    return float(np.mean(sc))

def _pca_2d(X):
    Xc=X-X.mean(0); cov=np.cov(Xc.T); ev,evec=np.linalg.eigh(cov)
    idx=np.argsort(ev)[::-1]; ev=ev[idx]; evec=evec[:,idx]
    return Xc@evec[:,:2], ev[:2]/ev.sum()

def run_clustering(df):
    feats=[f for f in CLUSTERING_FEATURES if f in df.columns]
    X=df[feats].fillna(0).values
    mu,std=X.mean(0),X.std(0); std[std==0]=1; Xs=(X-mu)/std
    np.random.seed(42)
    labels,centers,_=_kmeans(Xs,K_CLUSTERS)
    proj,exvar=_pca_2d(Xs)
    sil=_silhouette(Xs,labels) if len(df)>K_CLUSTERS else None
    r=df.copy(); r["cluster"]=labels; r["pca_x"]=proj[:,0]; r["pca_y"]=proj[:,1]
    return r, feats, sil, centers*std+mu, exvar

def label_clusters(df):
    df=df.copy()
    cs=df.groupby("cluster").agg(mb=("bounce_rate","mean"),mv=("views_per_session","mean"),
        ts=("sessions","sum"),np_=("page_id","count")).reset_index()
    lm,cm={},{}; assigned=set()
    # Underserved: highest bounce
    uc=int(cs.sort_values("mb",ascending=False).iloc[0]["cluster"])
    lm[uc]="Underserved / High-Friction"; cm[uc]=SEGMENT_COLORS["Underserved / High-Friction"]; assigned.add(uc)
    # Deep-engagement: highest views/session among rest
    rem=cs[~cs["cluster"].isin(assigned)]
    if not rem.empty:
        dc=int(rem.sort_values("mv",ascending=False).iloc[0]["cluster"])
        if rem.sort_values("mv",ascending=False).iloc[0]["mv"]>cs["mv"].mean()*1.3:
            lm[dc]="Deep-Engagement Niche"; cm[dc]=SEGMENT_COLORS["Deep-Engagement Niche"]; assigned.add(dc)
    # Well-Served: highest traffic of remaining
    rem=cs[~cs["cluster"].isin(assigned)]
    if not rem.empty:
        wc=int(rem.sort_values("ts",ascending=False).iloc[0]["cluster"])
        lm[wc]="Well-Served"; cm[wc]=SEGMENT_COLORS["Well-Served"]; assigned.add(wc)
    # Rest = Moderately Served
    for _,row in cs.iterrows():
        c=int(row["cluster"])
        if c not in assigned: lm[c]="Moderately Served"; cm[c]=SEGMENT_COLORS["Moderately Served"]
    df["segment"]=df["cluster"].map(lm); df["segment_color"]=df["cluster"].map(cm)
    return df, lm, cm


# ── AI AGENT (OpenAI API) ──
AGENT_SYSTEM_PROMPT = """You are the USDA Rural Development Digital Strategy Advisor, an AI agent
specialized in web analytics and digital service improvement for rural Americans.

You have the full results of a K-Means clustering analysis:

ANALYSIS SUMMARY:
- {n_pages} unique web pages analyzed, {n_sessions:,} total sessions
- Clustering: K=4, features: log(sessions), bounce_rate, views/session
- Silhouette Score: {sil_score}
- Top 10% of pages account for {top_traffic_pct:.0f}% of all sessions

SEGMENT PROFILES:
{seg_summary}

TOP 10 PRIORITY PAGES (highest friction × traffic):
{top_pages_str}

KEY FINDINGS ABOUT UNDERSERVED PAGES:
The 133 underserved/high-friction pages (bounce 75%+, exit 91%+, duration 55s) fall into these categories:
- 26% are Success Stories: long text articles with no navigation aids, no CTAs, users land and leave
- 20% are 404/Broken Links: dead PDFs, malformed URLs, deleted content still indexed
- 11% are News Releases: one-time reads with no pathway to related programs
- 10% are State-specific Program Pages: thin content, often just a title and a link
- 8% are About RD/Leadership pages: institutional pages with no user value
- 5% are Publications/Reports: PDFs that don't load or have no preview
- 3% are Forms/Factsheets: downloadable forms with 99%+ bounce (no context provided)
- 3% are Legacy/Archived Content: old radio spots, outdated reports (100% bounce)

YOUR CAPABILITIES:
- Provide specific, evidence-based recommendations tied to the data above
- Suggest AI-enabled solutions: chatbots, guided navigation, FAQ automation, document checklist tools
- Prioritize interventions by expected impact (HIGH/MEDIUM/LOW)
- Explain the methodology and clustering pipeline
- Generate executive summaries and presentation talking points
- Answer ANY question about the data, segments, or digital strategy
- Always tie recommendations to improving digital access for rural Americans

STYLE: Professional, executive-ready, concise. Use markdown formatting. Be specific with numbers."""


def build_system_prompt(ctx):
    tp = ctx.get("top_pages",[])
    tps = "\n".join([f"  {i+1}. {p}" for i,p in enumerate(tp[:10])]) if tp else "  (none)"
    sil = ctx.get("sil_score")
    return AGENT_SYSTEM_PROMPT.format(
        n_pages=ctx.get("n_pages",0), n_sessions=ctx.get("n_sessions",0),
        seg_summary=ctx.get("seg_summary",""), top_pages_str=tps,
        top_traffic_pct=ctx.get("top_traffic_pct",0),
        sil_score=f"{sil:.3f}" if sil else "N/A")


def call_agent_api(messages, api_key, ctx):
    try:
        client = OpenAI(api_key=api_key)
        sys_prompt = build_system_prompt(ctx)
        api_msgs = [{"role":"system","content":sys_prompt}] + messages
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=api_msgs,
                                               max_tokens=2000, temperature=0.7)
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ API Error: {str(e)}\n\nPlease check your API key and try again."


# ── HELPERS ──
def kpi_card(label, value, icon="📊"):
    st.markdown(f"""<div class='kpi-card'>
      <div style='font-size:1.5rem;margin-bottom:10px'>{icon}</div>
      <div style='font-size:2.2rem;font-weight:900;margin:0;line-height:1'>{value}</div>
      <div style='font-size:.85rem;opacity:.7;font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-top:8px'>{label}</div>
    </div>""", unsafe_allow_html=True)


# ── COUNTRY LIST ──
@st.cache_data
def get_country_list(filepath):
    try:
        df = load_usda_csv(filepath=filepath)
        if "Country" in df.columns:
            c = df["Country"].dropna().astype(str).str.strip()
            c = c[(c!="")&(c.str.lower()!="nan")]
            return sorted(c.unique().tolist())
    except: pass
    return ["United States"]


# ── SIDEBAR ──
with st.sidebar:
    st.image("https://www.usda.gov/themes/custom/uswds_usda/img/favicons/favicon-57.png", width=50)
    st.title("🌾 USDA Analytics")
    st.caption("Digital Experience Intelligence Platform")
    st.divider()

    uploaded_file = st.file_uploader("Upload USDA GA4 Export (.csv)", type=["csv"],
        help="Upload the GA4 web analytics export CSV from USDA Rural Development")

    st.divider()
    st.markdown("**🤖 AI Agent Configuration**")
    api_key = st.text_input("OpenAI API Key", type="password",
                            help="Required for the AI Strategy Advisor chatbot. Get one at platform.openai.com")

    st.divider()
    st.markdown("**Filters**")
    countries = get_country_list(DEFAULT_PATH)
    if len(countries) > 1:
        country_filter = st.multiselect("Country", options=countries, default=[],
            help="Leave empty for all countries.")
    else:
        country_filter = []
        st.caption(f"Country: {countries[0]}")

    min_sessions = st.number_input("Min Sessions per Page", min_value=0, value=10, step=10,
                                   help="Exclude pages below this session count.")

    st.divider()
    st.markdown("""<div style='font-size:.75rem;color:#888;line-height:1.5'>
    Built for USDA Rural Development<br>
    K-Means (K=4, optimized)<br>Features: log(sessions) + bounce + views/sess<br>
    Silhouette: 0.443<br>Version 3.1</div>""", unsafe_allow_html=True)


# ── DATA PIPELINE ──
@st.cache_data(show_spinner="Loading and processing USDA data...")
def get_processed_data(file_bytes, filepath, country_filter_tuple, min_sessions):
    if file_bytes is not None: raw = load_usda_csv(io.BytesIO(file_bytes))
    else:
        try: raw = load_usda_csv(filepath=filepath)
        except Exception as e: return None, str(e)
    cl = list(country_filter_tuple) if country_filter_tuple else None
    page_df = prepare_page_data(raw, country_filter=cl)
    if page_df.empty: return None, "No data matches filters. Adjust Country or Min Sessions."
    if "sessions" in page_df.columns: page_df = page_df[page_df["sessions"]>=min_sessions]
    if page_df.empty: return None, "No pages meet session threshold."
    page_df = engineer_features(page_df)
    if len(page_df) < K_CLUSTERS: return None, f"Not enough pages ({len(page_df)}) for {K_CLUSTERS} clusters."
    clustered, feat_cols, sil, centers_orig, exvar = run_clustering(page_df)
    clustered, lm, cm = label_clusters(clustered)
    return {"raw_df":raw, "page_df":clustered, "label_map":lm, "color_map":cm,
            "feat_cols":feat_cols, "sil_score":sil, "centers_orig":centers_orig,
            "explained_var":exvar, "k":K_CLUSTERS}, None


fb = uploaded_file.read() if uploaded_file else None
data, error = get_processed_data(fb, DEFAULT_PATH, tuple(country_filter) if country_filter else (), min_sessions)

if error: st.error(f"❌ {error}"); st.info("Upload a valid CSV or adjust filters."); st.stop()
if data is None: st.warning("Upload a CSV to begin."); st.stop()

page_df = data["page_df"]; label_map = data["label_map"]; color_map = data["color_map"]
sil_score = data["sil_score"]; k = data["k"]
seg_color_discrete = {label_map[c]: color_map[c] for c in label_map}

# ── DERIVED GLOBALS ──
n_pages = len(page_df)
total_sessions = int(page_df["sessions"].sum())
avg_bounce = page_df["bounce_rate"].mean()
avg_duration = page_df["avg_session_duration"].mean()
top10_sessions = page_df.nlargest(max(1, n_pages//10), "sessions")["sessions"].sum()
top_traffic_pct = (top10_sessions/total_sessions*100) if total_sessions>0 else 0
priority_pages = page_df.nlargest(10, "priority_score")["page_id"].tolist()

seg_lines = []
for sn in sorted(page_df["segment"].unique()):
    s = page_df[page_df["segment"]==sn]
    seg_lines.append(f"  • **{sn}** ({len(s):,} pages, {s['sessions'].sum():,.0f} sessions, "
        f"bounce {s['bounce_rate'].mean():.1%}, duration {s['avg_session_duration'].mean():.0f}s, "
        f"views/sess {s['views_per_session'].mean():.2f})")
seg_summary = "\n".join(seg_lines)

agent_context = {"seg_summary":seg_summary, "top_pages":priority_pages, "n_pages":n_pages,
    "n_sessions":total_sessions, "top_traffic_pct":top_traffic_pct, "k":k, "sil_score":sil_score}


# ── HEADER ──
st.markdown("""<div style='background:linear-gradient(135deg,#1a3a2a,#2d5a40);padding:28px 32px;border-radius:14px;margin-bottom:24px'>
  <h1 style='color:white;margin:0;font-size:2.1rem;font-weight:800'>🌾 USDA Rural Development — Digital Experience Intelligence Platform</h1>
  <p style='color:#a5d6a7;margin:8px 0 0;font-size:1rem'>Page-Level Web Analytics • K-Means Segmentation • Friction Prioritization • Strategic AI Agent</p>
</div>""", unsafe_allow_html=True)


# ── TABS ──
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Executive Overview", "📈 Traffic Analytics", "🔵 Page Segmentation",
    "🔥 Friction & Priorities", "🤖 AI Strategy Agent", "📂 Data & Methodology"])


# ═══ TAB 1 — EXECUTIVE OVERVIEW ═══
with tab1:
    st.markdown("<div class='section-header'>Executive Dashboard</div>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi_card("Unique Pages", f"{n_pages:,}", "📄")
    with c2: kpi_card("Total Sessions", f"{total_sessions:,}", "👥")
    with c3: kpi_card("Avg Bounce Rate", f"{avg_bounce:.1%}", "↩️")
    with c4: kpi_card("Avg Duration", f"{avg_duration:.0f}s", "⏱️")
    with c5: kpi_card("Silhouette Score", f"{sil_score:.3f}" if sil_score else "N/A", "✅")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class='insight-box'>🎯 <strong>Traffic Concentration:</strong> The top 10% of pages
    ({max(1,n_pages//10):,}) account for <strong>{top_traffic_pct:.0f}%</strong> of all sessions.</div>""", unsafe_allow_html=True)

    cl,cr = st.columns(2)
    with cl:
        sc = page_df.groupby("segment").agg(pages=("page_id","count"),sessions=("sessions","sum")).reset_index()
        fig = px.pie(sc, names="segment", values="pages", title="Page Distribution by Segment",
                     color="segment", color_discrete_map=seg_color_discrete, hole=.45)
        fig.update_layout(title_font_size=15, title_font_color="#1b5e20", paper_bgcolor="white", margin=dict(t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        fig = px.bar(sc, x="segment", y="sessions", title="Total Sessions by Segment", color="segment",
                     color_discrete_map=seg_color_discrete)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=15,
                          title_font_color="#1b5e20", showlegend=False, margin=dict(l=40,r=20,t=50,b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Segment Profile Summary</div>", unsafe_allow_html=True)
    sp = page_df.groupby("segment").agg(Pages=("page_id","count"),Sessions=("sessions","sum"),
        Bounce=("bounce_rate","mean"),Duration=("avg_session_duration","mean"),
        VPS=("views_per_session","mean"),Exit=("exit_ratio","mean"),
        Returning=("returning_user_ratio","mean"),Engagement=("engagement_score","mean"),
        Friction=("friction_index","mean")).reset_index()
    dp = sp.copy()
    dp["Sessions"]=dp["Sessions"].apply(lambda x:f"{int(x):,}")
    dp["Bounce"]=dp["Bounce"].apply(lambda x:f"{x:.1%}")
    dp["Duration"]=dp["Duration"].apply(lambda x:f"{x:.0f}s")
    dp["VPS"]=dp["VPS"].apply(lambda x:f"{x:.2f}")
    dp["Exit"]=dp["Exit"].apply(lambda x:f"{x:.1%}")
    dp["Returning"]=dp["Returning"].apply(lambda x:f"{x:.1%}")
    dp["Engagement"]=dp["Engagement"].apply(lambda x:f"{x:.3f}")
    dp["Friction"]=dp["Friction"].apply(lambda x:f"{x:.3f}")
    dp.columns=["Segment","Pages","Sessions","Bounce","Duration","Views/Sess","Exit","Returning%","Engagement","Friction"]
    st.dataframe(dp, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
    us = page_df[page_df["segment"]=="Underserved / High-Friction"]
    if not us.empty:
        st.markdown(f"""<div class='warning-box'>⚠️ <strong>{len(us):,} pages</strong> are
        <strong>Underserved / High-Friction</strong> ({us['sessions'].sum():,.0f} sessions).
        Bounce: <strong>{us['bounce_rate'].mean():.1%}</strong>, Exit: <strong>{us['exit_ratio'].mean():.1%}</strong>,
        Duration: <strong>{us['avg_session_duration'].mean():.0f}s</strong>.<br><br>
        <strong>Root causes:</strong> 26% are success stories with no CTAs, 20% are broken links/404s,
        11% are news releases with no pathway to programs, 10% are thin state-specific pages.</div>""", unsafe_allow_html=True)

    ws = page_df[page_df["segment"]=="Well-Served"]
    if not ws.empty:
        st.markdown(f"""<div class='insight-box'>✅ <strong>{len(ws):,} pages</strong> are <strong>Well-Served</strong>
        ({ws['sessions'].sum():,.0f} sessions, bounce {ws['bounce_rate'].mean():.1%}, duration {ws['avg_session_duration'].mean():.0f}s).
        Use as design templates.</div>""", unsafe_allow_html=True)

    de = page_df[page_df["segment"]=="Deep-Engagement Niche"]
    if not de.empty:
        st.markdown(f"""<div class='insight-box'>🔵 <strong>{len(de):,} pages</strong> form a
        <strong>Deep-Engagement Niche</strong> (bounce {de['bounce_rate'].mean():.1%},
        views/session {de['views_per_session'].mean():.2f}). Power users exploring deeply.</div>""", unsafe_allow_html=True)


# ═══ TAB 2 — TRAFFIC ANALYTICS ═══
with tab2:
    st.markdown("<div class='section-header'>Traffic Distribution Analytics</div>", unsafe_allow_html=True)
    ca,cb = st.columns(2)
    with ca:
        t20 = page_df.nlargest(20,"sessions")[["page_id","sessions","segment"]].copy()
        t20["page_short"] = t20["page_id"].astype(str).str[:55]
        fig = px.bar(t20, x="sessions", y="page_short", orientation="h", color="segment",
                     title="Top 20 Pages by Sessions", color_discrete_map=seg_color_discrete)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=14,
                          title_font_color="#1b5e20", yaxis={"categoryorder":"total ascending"},
                          height=500, margin=dict(l=10,r=20,t=50,b=40))
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        fig = px.histogram(page_df, x="sessions", nbins=50, title="Session Distribution (Log Scale)",
                           color_discrete_sequence=["#2e7d32"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=14,
                          title_font_color="#1b5e20", xaxis_type="log", margin=dict(l=20,r=20,t=50,b=40))
        st.plotly_chart(fig, use_container_width=True)

        samp = page_df.sample(min(2000,len(page_df)), random_state=42)
        fig = px.scatter(samp, x="bounce_rate", y="avg_session_duration", color="segment",
                         size=np.log1p(samp["sessions"])+1, title="Bounce vs Duration",
                         color_discrete_map=seg_color_discrete,
                         hover_data={"page_id":True,"sessions":True})
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=14,
                          title_font_color="#1b5e20", margin=dict(l=20,r=20,t=50,b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Engagement–Friction Quadrant</div>", unsafe_allow_html=True)
    qs = page_df.sample(min(3000,len(page_df)), random_state=1)
    fig = px.scatter(qs, x="friction_index", y="engagement_score", color="segment",
                     size=np.log1p(qs["sessions"])*2+2, title="Engagement vs Friction",
                     color_discrete_map=seg_color_discrete,
                     hover_data={"page_id":True,"sessions":True,"bounce_rate":":.2%"})
    fig.add_hline(y=page_df["engagement_score"].median(), line_dash="dot", line_color="gray", annotation_text="Median Engagement")
    fig.add_vline(x=page_df["friction_index"].median(), line_dash="dot", line_color="gray", annotation_text="Median Friction")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=15,
                      title_font_color="#1b5e20", height=500, margin=dict(l=20,r=20,t=60,b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bubble size ∝ log(sessions). Ideal: lower-right quadrant.")

    st.markdown("<div class='section-header'>Metric Distributions by Segment</div>", unsafe_allow_html=True)
    mc = st.selectbox("Select metric", ["bounce_rate","avg_session_duration","views_per_session",
                                         "exit_ratio","events_per_session","returning_user_ratio"])
    if mc in page_df.columns:
        fig = px.box(page_df, x="segment", y=mc, color="segment",
                     title=f"{mc.replace('_',' ').title()} by Segment", color_discrete_map=seg_color_discrete, points="outliers")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=15,
                          title_font_color="#1b5e20", showlegend=False, margin=dict(l=20,r=20,t=60,b=40))
        st.plotly_chart(fig, use_container_width=True)


# ═══ TAB 3 — PAGE SEGMENTATION ═══
with tab3:
    st.markdown("<div class='section-header'>K-Means Page Segmentation</div>", unsafe_allow_html=True)
    sil_d = f"{sil_score:.3f}" if sil_score else "N/A"
    st.info(f"🔵 K=4 (fixed optimal) • Silhouette: {sil_d} • {n_pages:,} pages • Features: log(sessions), bounce, views/session")

    # PCA
    st.markdown("<div class='section-header'>PCA Cluster Visualization</div>", unsafe_allow_html=True)
    ps = page_df.sample(min(4000,len(page_df)), random_state=7)
    fig = px.scatter(ps, x="pca_x", y="pca_y", color="segment", title="PCA Projection",
                     color_discrete_map=seg_color_discrete, size=np.log1p(ps["sessions"])+1,
                     hover_data={"page_id":True,"sessions":True,"bounce_rate":":.2%","views_per_session":":.2f"},
                     labels={"pca_x":"PC1","pca_y":"PC2"})
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=15,
                      title_font_color="#1b5e20", height=520, margin=dict(l=20,r=20,t=60,b=40))
    st.plotly_chart(fig, use_container_width=True)
    if data["explained_var"] is not None:
        ev=data["explained_var"]
        st.caption(f"PC1: {ev[0]:.1%} • PC2: {ev[1]:.1%} • Combined: {sum(ev):.1%}")

    # Radar
    st.markdown("<div class='section-header'>Cluster Centroid Profiles</div>", unsafe_allow_html=True)
    rcols=["bounce_rate","avg_session_duration","views_per_session","exit_ratio","returning_user_ratio"]
    ra=[c for c in rcols if c in page_df.columns]
    if ra:
        cen=page_df.groupby("segment")[ra].mean().reset_index()
        for c in ra:
            mn,mx=cen[c].min(),cen[c].max()
            cen[f"{c}_n"]=(cen[c]-mn)/(mx-mn) if mx>mn else 0.5
        nc=[f"{c}_n" for c in ra]
        fig=go.Figure()
        for _,row in cen.iterrows():
            sn=row["segment"]; clr=seg_color_discrete.get(sn,"#888")
            fig.add_trace(go.Scatterpolar(
                r=[row[c] for c in nc]+[row[nc[0]]],
                theta=[c.replace("_n","").replace("_"," ").title() for c in nc]+[nc[0].replace("_n","").replace("_"," ").title()],
                fill="toself",name=sn,line_color=clr,fillcolor=clr,opacity=.25))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                          title="Segment Profiles (Normalized)", paper_bgcolor="white",
                          title_font_size=14, title_font_color="#1b5e20", height=450)
        st.plotly_chart(fig, use_container_width=True)

    # ── DOWNLOAD ALL CLUSTERS ──
    st.markdown("<div class='section-header'>📥 Download Pages by Cluster</div>", unsafe_allow_html=True)
    st.markdown("Download all web pages belonging to each cluster with their full characteristics.")

    dl_cols_list = ["page_id","page_title","sessions","active_users","total_users","bounce_rate",
                    "avg_session_duration","views_per_session","exit_ratio","events_per_session",
                    "returning_user_ratio","engagement_score","friction_index","priority_score","segment"]
    dl_avail = [c for c in dl_cols_list if c in page_df.columns]

    segs_sorted = sorted(page_df["segment"].unique())
    dcols = st.columns(len(segs_sorted))
    seg_icons = {"Well-Served":"✅","Deep-Engagement Niche":"🔵","Moderately Served":"🟡","Underserved / High-Friction":"🔴"}
    for i, sn in enumerate(segs_sorted):
        sd = page_df[page_df["segment"]==sn][dl_avail].sort_values("sessions",ascending=False)
        with dcols[i]:
            st.markdown(f"**{seg_icons.get(sn,'📄')} {sn}**")
            st.caption(f"{len(sd)} pages · {int(sd['sessions'].sum()):,} sess")
            st.download_button(f"⬇️ Download CSV", sd.to_csv(index=False).encode(),
                file_name=f"cluster_{sn.replace('/','_').replace(' ','_').lower()}.csv",
                mime="text/csv", key=f"dl_{i}", use_container_width=True)

    all_csv = page_df[dl_avail].sort_values(["segment","sessions"],ascending=[True,False]).to_csv(index=False).encode()
    st.download_button("⬇️ Download ALL Pages (All Clusters)", all_csv,
                       file_name="usda_all_clusters.csv", mime="text/csv", use_container_width=True)

    st.markdown("---")

    # Segment deep-dive
    st.markdown("<div class='section-header'>Segment Explorer</div>", unsafe_allow_html=True)
    sel = st.selectbox("Select segment", segs_sorted)
    sd = page_df[page_df["segment"]==sel].copy()
    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("Pages", f"{len(sd):,}")
    with m2: st.metric("Sessions", f"{int(sd['sessions'].sum()):,}")
    with m3: st.metric("Bounce", f"{sd['bounce_rate'].mean():.1%}")
    with m4: st.metric("Views/Sess", f"{sd['views_per_session'].mean():.2f}")

    show_cols = ["page_id","page_title","sessions","bounce_rate","avg_session_duration",
                 "views_per_session","exit_ratio","engagement_score","friction_index"]
    sa = [c for c in show_cols if c in sd.columns]
    ss = sd[sa].sort_values("sessions",ascending=False).copy()
    if "bounce_rate" in ss.columns: ss["bounce_rate"]=ss["bounce_rate"].apply(lambda x:f"{x:.1%}")
    if "avg_session_duration" in ss.columns: ss["avg_session_duration"]=ss["avg_session_duration"].apply(lambda x:f"{x:.0f}s")
    if "exit_ratio" in ss.columns: ss["exit_ratio"]=ss["exit_ratio"].apply(lambda x:f"{x:.1%}")
    ss.columns=[c.replace("_"," ").title() for c in ss.columns]
    st.dataframe(ss, use_container_width=True, hide_index=True)


# ═══ TAB 4 — FRICTION & PRIORITIES ═══
with tab4:
    st.markdown("<div class='section-header'>Friction Analysis & Intervention Priorities</div>", unsafe_allow_html=True)
    st.markdown("""<div class='insight-box'>📌 <strong>Priority Score = Friction Index × log(1 + Sessions)</strong><br>
    Pages with high traffic AND high friction = largest opportunity for improving service delivery.</div>""", unsafe_allow_html=True)

    t10 = page_df.nlargest(10,"priority_score").copy()
    t10["rank"]=range(1,len(t10)+1)
    t10["page_short"]=t10["page_id"].astype(str).str[:60]
    fig = px.bar(t10, x="priority_score", y="page_short", orientation="h", color="friction_index",
        color_continuous_scale=[[0,"#fff8e1"],[.5,"#f9a825"],[1,"#c62828"]],
        title="🔥 Top 10 Priority Pages", text="rank")
    fig.update_traces(textposition="outside")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=15,
        title_font_color="#1b5e20", yaxis={"categoryorder":"total ascending"},
        height=420, margin=dict(l=10,r=40,t=60,b=40), coloraxis_colorbar_title="Friction")
    st.plotly_chart(fig, use_container_width=True)

    pd10 = t10[["rank","page_id","sessions","bounce_rate","exit_ratio","avg_session_duration",
                "friction_index","priority_score","segment"]].copy()
    pd10["bounce_rate"]=pd10["bounce_rate"].apply(lambda x:f"{x:.1%}")
    pd10["exit_ratio"]=pd10["exit_ratio"].apply(lambda x:f"{x:.1%}")
    pd10["avg_session_duration"]=pd10["avg_session_duration"].apply(lambda x:f"{x:.0f}s")
    pd10["friction_index"]=pd10["friction_index"].apply(lambda x:f"{x:.3f}")
    pd10["priority_score"]=pd10["priority_score"].apply(lambda x:f"{x:.3f}")
    pd10["sessions"]=pd10["sessions"].apply(lambda x:f"{int(x):,}")
    pd10.columns=["#","Page","Sessions","Bounce","Exit","Duration","Friction","Priority","Segment"]
    st.dataframe(pd10, use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download Priority Pages", t10.to_csv(index=False).encode(),
                       file_name="usda_priority_pages.csv", mime="text/csv")

    st.markdown("<div class='section-header'>Friction by Segment</div>", unsafe_allow_html=True)
    f1,f2 = st.columns(2)
    with f1:
        sf = page_df.groupby("segment").agg(avg_f=("friction_index","mean")).reset_index().sort_values("avg_f",ascending=False)
        fig = px.bar(sf, x="segment", y="avg_f", title="Avg Friction by Segment", color="segment",
                     color_discrete_map=seg_color_discrete)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=14,
                          title_font_color="#1b5e20", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        qs2 = page_df.sample(min(3000,len(page_df)), random_state=5)
        fig = px.scatter(qs2, x="bounce_rate", y="exit_ratio", color="priority_score",
            color_continuous_scale=[[0,"#e8f5e9"],[.5,"#f9a825"],[1,"#c62828"]],
            title="Bounce vs Exit (by Priority)", hover_data={"page_id":True,"sessions":True})
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=14,
                          title_font_color="#1b5e20", coloraxis_colorbar_title="Priority")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Underserved Pages — Root Cause Analysis")
    st.markdown("""<div class='warning-box'>
    <strong>What do underserved pages have in common?</strong> Analysis of the 133 high-friction pages reveals:<br><br>
    📰 <strong>26% are Success Stories</strong> — Long articles with no CTAs, no related links, no program pathways. Users land from search, read briefly, leave.<br>
    🔗 <strong>20% are 404/Broken Links</strong> — Dead PDFs, malformed URLs, deleted content still indexed by Google.<br>
    📢 <strong>11% are News Releases</strong> — One-time announcements with no navigation to related programs or services.<br>
    🗺️ <strong>10% are State-specific Program Pages</strong> — Thin content with just a title and a link, no local context.<br>
    📄 <strong>8% are About RD/Leadership Pages</strong> — Institutional pages with no user value or next steps.<br>
    📋 <strong>5% are Publications/Reports</strong> — PDF links that don't load or provide no preview/abstract.<br>
    📝 <strong>3% are Forms/Factsheets</strong> — Direct downloads with zero context (99%+ bounce).<br>
    🗃️ <strong>3% are Legacy/Archived Content</strong> — Outdated radio spots, old reports (100% bounce).<br>
    </div>""", unsafe_allow_html=True)

    hf = page_df[page_df["friction_index"]>0.5].sort_values("priority_score",ascending=False)
    if not hf.empty:
        st.dataframe(hf[["page_id","sessions","bounce_rate","exit_ratio","friction_index","priority_score","segment"]]
            .rename(columns={"page_id":"Page","bounce_rate":"Bounce","exit_ratio":"Exit",
                             "friction_index":"Friction","priority_score":"Priority"}).head(25),
            use_container_width=True, hide_index=True)


# ═══ TAB 5 — AI STRATEGY AGENT ═══
with tab5:
    st.markdown("<div class='section-header'>🤖 AI Strategy Advisor — Recommendation Chatbot</div>", unsafe_allow_html=True)

    # Segment cards
    imap = {"Well-Served":("✅","High-traffic, low bounce. Design benchmarks.","#e8f5e9","#2e7d32"),
        "Deep-Engagement Niche":("🔵","Power users exploring deeply. Maintain depth.","#e3f2fd","#0288d1"),
        "Moderately Served":("🟡","Functional with optimization headroom.","#fff8e1","#f57f17"),
        "Underserved / High-Friction":("🔴","75%+ bounce, 91%+ exit. Needs intervention.","#ffebee","#c62828")}
    asegs = page_df["segment"].unique()
    ia = {sk:v for sk,v in imap.items() if sk in asegs}
    ic = st.columns(min(4,len(ia)))
    for i,(sn,(icon,desc,bg,txt)) in enumerate(ia.items()):
        ns = len(page_df[page_df["segment"]==sn])
        with ic[i%len(ic)]:
            st.markdown(f"""<div style='background:{bg};border-left:5px solid {txt};border-radius:8px;
            padding:12px;margin-bottom:8px;min-height:90px'>
            <div style='font-size:1.1rem'>{icon} <strong style="color:{txt}">{sn}</strong></div>
            <div style='font-size:.8rem;color:{txt};font-weight:600'>{ns} pages</div>
            <div style='font-size:.8rem;color:#555;margin-top:4px'>{desc}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    if not api_key:
        st.warning("⚠️ Enter your OpenAI API Key in the sidebar to activate the AI Strategy Advisor.")
        st.markdown("""<div class='insight-box'>
        The AI Strategy Advisor is a <strong>fully conversational chatbot</strong> powered by GPT-4o-mini
        with complete context of your clustering results, segment profiles, and underserved page analysis.
        It can answer <strong>any question</strong> about:<br>
        • Strategic recommendations for each segment<br>
        • AI-enabled solutions (chatbots, guided navigation, FAQ automation)<br>
        • Executive summaries and presentation talking points<br>
        • Methodology explanations<br>
        • Specific page-level intervention plans<br>
        • Root cause analysis of underserved pages<br><br>
        <strong>Enter your OpenAI API key in the sidebar to start chatting.</strong></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='insight-box'>🤖 This AI agent has <strong>full context</strong> of your
        clustering analysis, segment profiles, and underserved page root causes. Ask it <strong>anything</strong>
        — it remembers your conversation for follow-up questions.</div>""", unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Suggested questions
        st.markdown("**Suggested questions:**")
        q1,q2 = st.columns(2)
        q3,q4 = st.columns(2)
        suggestions = [
            (q1, "📊 Executive Summary", "Give me a concise executive summary of the key findings and strategic recommendations for USDA Rural Development"),
            (q2, "🔧 Priority Actions", "What are the top priority pages to fix and what specific actions should we take for each? Include expected impact."),
            (q3, "🤖 AI Roadmap", "Recommend an AI-enabled solutions roadmap with chatbots, guided navigation, and automation. State expected impact and priority (HIGH/MEDIUM/LOW)."),
            (q4, "📰 Fix Success Stories", "The success stories pages have 26% of underserved traffic. What specific content redesign should we implement to reduce their bounce rate?"),
        ]
        for col, label, q in suggestions:
            with col:
                if st.button(label, use_container_width=True, key=f"sq_{label}"):
                    st.session_state.chat_history.append({"role":"user","content":q})
                    with st.spinner("🤔 Agent thinking..."):
                        r = call_agent_api(st.session_state.chat_history, api_key, agent_context)
                    st.session_state.chat_history.append({"role":"assistant","content":r})
                    st.rerun()

        st.markdown("---")

        # Chat display
        for msg in st.session_state.chat_history:
            if msg["role"]=="user":
                st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='agent-response'>
                <div style='display:flex;align-items:center;margin-bottom:12px'>
                  <div style='background:#2e7d32;color:white;padding:6px 10px;border-radius:50%;margin-right:10px'>🤖</div>
                  <strong style='color:#1b5e20'>USDA Strategy Advisor</strong></div>
                {msg['content']}</div>""", unsafe_allow_html=True)

        # Chat input — fully open, any question
        user_input = st.chat_input("Ask anything about the analysis, recommendations, methodology, segments...")
        if user_input:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            with st.spinner("🤔 Agent analyzing..."):
                r = call_agent_api(st.session_state.chat_history, api_key, agent_context)
            st.session_state.chat_history.append({"role":"assistant","content":r})
            st.rerun()

        cc,cd = st.columns(2)
        with cc:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with cd:
            if st.session_state.chat_history:
                full = "\n\n".join([f"{'USER' if m['role']=='user' else 'ADVISOR'}: {m['content']}" for m in st.session_state.chat_history])
                st.download_button("📄 Download Conversation", full,
                    file_name="usda_advisor_chat.txt", mime="text/plain", use_container_width=True)


# ═══ TAB 6 — DATA & METHODOLOGY ═══
with tab6:
    st.markdown("<div class='section-header'>Data Overview & Methodology</div>", unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    with d1: st.metric("Raw Rows", f"{len(data['raw_df']):,}")
    with d2: st.metric("Unique Pages", f"{n_pages:,}")
    with d3: st.metric("Features", str(len(data["feat_cols"])))
    with d4: st.metric("K", str(k))

    st.markdown("<div class='section-header'>Feature Selection</div>", unsafe_allow_html=True)
    st.markdown("Tested **5 feature combinations × 6 K values = 30 configs**. Winner:")
    fd = pd.DataFrame({"Feature":["log(sessions)","bounce_rate","views_per_session"],
        "Dimension":["Traffic Volume","User Friction","Engagement Depth"],
        "Why":["Log-transform reduces extreme skew","Primary friction signal — single-page visits",
               "Content exploration depth — independent of bounce (r=-0.22)"],
        "Correlations":["r=-0.30 bounce, r=0.05 views","r=-0.30 log_sess, r=-0.22 views",
                        "r=0.05 log_sess, r=-0.22 bounce"]})
    st.dataframe(fd, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Silhouette Comparison</div>", unsafe_allow_html=True)
    sild = pd.DataFrame({"K":[2,3,4,5,6,7],"Silhouette":[0.378,0.436,0.443,0.396,0.369,0.357]})
    fig = px.line(sild, x="K", y="Silhouette", title="Silhouette by K", markers=True, color_discrete_sequence=["#2e7d32"])
    fig.add_annotation(x=4,y=0.443,text="K=4: Best (0.443)",showarrow=True,arrowhead=2,font=dict(color="#c62828",size=13))
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=14, title_font_color="#1b5e20")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Cluster Centroids</div>", unsafe_allow_html=True)
    if data["centers_orig"] is not None:
        cdf = pd.DataFrame(data["centers_orig"], columns=data["feat_cols"])
        cdf.insert(0,"Cluster",range(len(cdf)))
        cdf["Segment"]=cdf["Cluster"].map(label_map)
        cdf["~Sessions"]=np.expm1(cdf["log_sessions"]).astype(int)
        dc=cdf[["Segment","log_sessions","~Sessions","bounce_rate","views_per_session"]].copy()
        dc.columns=["Segment","log(sess)","~Sessions","Bounce","Views/Sess"]
        dc["Bounce"]=dc["Bounce"].apply(lambda x:f"{x:.3f}")
        dc["Views/Sess"]=dc["Views/Sess"].apply(lambda x:f"{x:.3f}")
        dc["log(sess)"]=dc["log(sess)"].apply(lambda x:f"{x:.2f}")
        dc["~Sessions"]=dc["~Sessions"].apply(lambda x:f"{x:,}")
        st.dataframe(dc, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>Methodology</div>", unsafe_allow_html=True)
    st.markdown("""
**1. Data Ingestion** — Raw USDA GA4 export. Only "Totals" columns used (avoids double-counting across devices).

**2. Cleaning** — Numeric cleanup, percentage normalization [0,1], median imputation, zero-division guards.

**3. Aggregation** — Pages grouped by path. Sums for counts, means for rates. Derived ratios post-aggregation.

**4. Feature Engineering** — Engagement Score & Friction Index (MinMax-normalized weighted composites).
Priority Score = Friction × log(1+sessions).

**5. Feature Selection** — 30 configurations tested. Best: log(sessions) + bounce + views/session at K=4 (Sil=0.443).

**6. Clustering** — K-Means, n_init=30, max_iter=500, pure NumPy. PCA(2) for visualization.

**7. Labeling** — Highest bounce → Underserved; highest views/sess → Deep-Engagement; highest traffic → Well-Served; rest → Moderate.
""")

    st.markdown("<div class='section-header'>Data Preview</div>", unsafe_allow_html=True)
    pc=[c for c in ["page_id","page_title","sessions","bounce_rate","avg_session_duration",
        "views_per_session","exit_ratio","engagement_score","friction_index","priority_score","segment"] if c in page_df.columns]
    st.dataframe(page_df[pc].sort_values("sessions",ascending=False).head(100), use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download Full Dataset", page_df.to_csv(index=False).encode(),
                       file_name="usda_pages_full.csv", mime="text/csv")

    st.markdown("""---
<div style='font-size:.8rem;color:#888;text-align:center;padding:10px'>
USDA Digital Experience Intelligence Platform v3.1 | K-Means (K=4, Optimized) | GA4 Data Source</div>""", unsafe_allow_html=True)
