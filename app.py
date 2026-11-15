from __future__ import annotations
import io, os, zipfile, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="AIRGUARD INDIA | Air Quality Intelligence",
                   page_icon="🌫️", layout="wide", initial_sidebar_state="expanded")
CSS_PATH = Path(__file__).parent / "assets" / "custom.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "city_day.csv"
POLLUTANTS = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]
AQI_ORDER = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
COLORS = ["#22d3ee","#818cf8","#a78bfa","#f59e0b","#fb7185","#ef4444"]
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,17,35,.72)",
    font=dict(family="Inter, Arial, sans-serif", color="#dbeafe", size=12),
    margin=dict(l=18,r=18,t=48,b=18),
    xaxis=dict(gridcolor="rgba(148,163,184,.13)", zerolinecolor="rgba(148,163,184,.15)"),
    yaxis=dict(gridcolor="rgba(148,163,184,.13)", zerolinecolor="rgba(148,163,184,.15)"),
    legend=dict(bgcolor="rgba(0,0,0,0)")
)

def styled(fig, height=360):
    fig.update_layout(**PLOT_LAYOUT, height=height, hoverlabel=dict(bgcolor="#101b32", font_color="#e2e8f0"))
    return fig

def load_csv(source):
    if source is None:
        return None
    try:
        if isinstance(source, (str, Path)):
            return pd.read_csv(source)
        raw = source.getvalue()
        if source.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                match = next((n for n in z.namelist() if n.lower().endswith("city_day.csv")), None)
                if not match: raise ValueError("ZIP does not contain city_day.csv.")
                with z.open(match) as f: return pd.read_csv(f)
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        st.error(f"Could not read the dataset: {exc}")
        return None

def normalize_data(raw):
    if raw is None or raw.empty:
        return None
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # tolerate common case / whitespace variations
    lookup = {c.lower().replace(" ", "").replace("_",""): c for c in df.columns}
    ren = {}
    for canonical in ["City","Date","AQI","AQI_Bucket"] + POLLUTANTS:
        key = canonical.lower().replace("_","").replace(" ","")
        if key in lookup and lookup[key] != canonical: ren[lookup[key]] = canonical
    df = df.rename(columns=ren)
    required = ["City","Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Required column(s) missing: " + ", ".join(missing))
    df["City"] = df["City"].astype("string").str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["City","Date"]).copy()
    for c in POLLUTANTS + ["AQI"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "AQI_Bucket" in df.columns:
        df["AQI_Bucket"] = df["AQI_Bucket"].astype("string").str.strip().replace({"":pd.NA, "nan":pd.NA})
    df = df.sort_values(["City","Date"]).reset_index(drop=True)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["Quarter"] = "Q" + df["Date"].dt.quarter.astype(str)
    df["Season"] = df["Month"].map({12:"Winter",1:"Winter",2:"Winter",3:"Summer",4:"Summer",5:"Summer",
                                    6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Post-monsoon",10:"Post-monsoon",11:"Post-monsoon"})
    return df

def metric_card(label, value, note=""):
    st.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>""", unsafe_allow_html=True)

def empty_chart(message="Not enough valid observations for this view."):
    st.info(message)

def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan

def filter_controls(df):
    st.sidebar.markdown("### ◈ Analysis filters")
    cities = sorted(df["City"].dropna().astype(str).unique())
    selected_cities = st.sidebar.multiselect("Cities", cities, default=cities, key="cities_filter")
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_d,max_d), min_value=min_d, max_value=max_d, key="date_filter")
    buckets = sorted(df["AQI_Bucket"].dropna().astype(str).unique()) if "AQI_Bucket" in df else []
    selected_buckets = st.sidebar.multiselect("AQI categories", buckets, default=buckets, key="bucket_filter")
    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    else:
        start, end = pd.Timestamp(min_d), pd.Timestamp(max_d) + pd.Timedelta(days=1)
    out = df[df["City"].isin(selected_cities) & df["Date"].between(start,end)].copy()
    if "AQI_Bucket" in out and selected_buckets:
        out = out[out["AQI_Bucket"].isin(selected_buckets)]
    elif "AQI_Bucket" in out and not selected_buckets:
        out = out.iloc[0:0]
    st.sidebar.caption(f"{len(out):,} rows after filters")
    if st.sidebar.button("↺ Reset filters", use_container_width=True):
        for key in ["cities_filter","date_filter","bucket_filter"]:
            st.session_state.pop(key, None)
        st.rerun()
    return out

def chart_or_info(series, chart_fn):
    if series is None or len(series.dropna()) == 0: empty_chart()
    else: st.plotly_chart(chart_fn(), use_container_width=True, config={"displayModeBar":True})

def render_overview(df, full):
    st.markdown("## Executive overview")
    st.caption("A national-level snapshot of the selected historical observations.")
    aq = df["AQI"].dropna() if "AQI" in df else pd.Series(dtype=float)
    cities = df["City"].nunique()
    pollutants_present = [p for p in POLLUTANTS if p in df.columns]
    coverage = f"{df.Date.min():%d %b %Y} — {df.Date.max():%d %b %Y}" if len(df) else "—"
    cols = st.columns(3)
    vals = [f"{len(df):,}", f"{cities:,}", f"{aq.mean():,.1f}" if len(aq) else "—",
            f"{aq.max():,.0f}" if len(aq) else "—", str(len(pollutants_present)), coverage]
    labels = ["Observations","Cities in view","Average AQI","Maximum AQI","Pollutants available","Date coverage"]
    notes = ["Filtered records","Distinct city names","Valid AQI values only","Uncapped observed maximum","Of 12 dataset pollutants","Selected date window"]
    for i in range(6):
        with cols[i%3]: metric_card(labels[i], vals[i], notes[i])
    st.markdown("### AQI distribution & city leaderboard")
    left,right = st.columns([1.05,1])
    with left:
        if len(aq):
            fig=px.histogram(df.dropna(subset=["AQI"]),x="AQI",nbins=45,color_discrete_sequence=["#22d3ee"],title="Observed AQI distribution")
            st.plotly_chart(styled(fig),use_container_width=True)
        else: empty_chart("AQI values are unavailable in this selection.")
    with right:
        if "AQI" in df and df["AQI"].notna().any():
            rank=df.groupby("City")["AQI"].agg(["mean","count"]).query("count > 0").sort_values("mean",ascending=False)
            fig=px.bar(rank.head(10).sort_values("mean"),x="mean",y=rank.head(10).sort_values("mean").index,orientation="h",
                       color="mean",color_continuous_scale=["#22d3ee","#818cf8","#fb7185"],title="10 highest city average AQI")
            fig.update_layout(coloraxis_showscale=False,xaxis_title="Average AQI",yaxis_title="")
            st.plotly_chart(styled(fig),use_container_width=True)
        else: empty_chart()
    if "AQI" in df and df["AQI"].notna().any():
        rank=df.groupby("City")["AQI"].mean().dropna().sort_values()
        c1,c2=st.columns(2)
        with c1:
            fig=px.bar(rank.head(10).sort_values(),x=rank.head(10).sort_values().values,y=rank.head(10).sort_values().index,
                       orientation="h",title="10 lowest city average AQI",color_discrete_sequence=["#34d399"])
            st.plotly_chart(styled(fig),use_container_width=True)
        with c2:
            mean=aq.mean(); med=aq.median(); top=rank.index[-1] if len(rank) else "—"
            st.markdown("#### Automated observations")
            st.markdown(f"- **{top}** has the highest mean AQI among cities in the current selection.")
            st.markdown(f"- Mean AQI is **{mean:.1f}** and median AQI is **{med:.1f}**.")
            st.markdown(f"- **{aq.isna().sum():,}** selected records have no AQI value.")
            st.markdown("- Averages are descriptive and may be affected by missing observations and extreme values.")

def render_city(df):
    st.markdown("## City intelligence")
    if "AQI" not in df or not df["AQI"].notna().any(): empty_chart("AQI column is absent or contains no valid values."); return
    summary=df.groupby("City").agg(avg_aqi=("AQI","mean"),median_aqi=("AQI","median"),records=("AQI","count")).dropna().sort_values("avg_aqi",ascending=False)
    l,r=st.columns([1.1,1])
    with l:
        fig=px.bar(summary.reset_index(),x="City",y="avg_aqi",color="avg_aqi",color_continuous_scale=["#22d3ee","#818cf8","#fb7185"],
                   hover_data=["median_aqi","records"],title="Average AQI by city")
        fig.update_layout(xaxis_tickangle=-40,coloraxis_showscale=False)
        st.plotly_chart(styled(fig,430),use_container_width=True)
    with r:
        st.markdown("### City ranking")
        st.dataframe(summary.rename(columns={"avg_aqi":"Average AQI","median_aqi":"Median AQI","records":"Valid AQI records"}).style.format({"Average AQI":"{:.1f}","Median AQI":"{:.1f}"}),use_container_width=True,height=410)
    selected=st.multiselect("Inspect city profiles",sorted(df.City.unique()),default=sorted(df.City.unique())[:min(3,df.City.nunique())],key="city_profile")
    if selected:
        sub=df[df.City.isin(selected)]
        fig=px.line(sub.dropna(subset=["AQI"]),x="Date",y="AQI",color="City",title="Daily AQI observations by selected city")
        st.plotly_chart(styled(fig,400),use_container_width=True)
        available=[p for p in POLLUTANTS if p in sub]
        if available:
            p=st.selectbox("Pollutant for city comparison",available,key="city_pollutant")
            ps=sub.groupby("City")[p].mean().dropna().sort_values(ascending=False).reset_index()
            fig=px.bar(ps,x="City",y=p,color="City",title=f"Mean {p} concentration by city")
            st.plotly_chart(styled(fig),use_container_width=True)
    if "AQI_Bucket" in df:
        ct=df.dropna(subset=["AQI_Bucket"]).groupby(["City","AQI_Bucket"]).size().reset_index(name="Observations")
        if len(ct):
            fig=px.bar(ct,x="City",y="Observations",color="AQI_Bucket",category_orders={"AQI_Bucket":AQI_ORDER},color_discrete_sequence=COLORS,
                       title="AQI category mix by city")
            fig.update_layout(barmode="stack",xaxis_tickangle=-40)
            st.plotly_chart(styled(fig,430),use_container_width=True)

def render_pollutants(df):
    st.markdown("## Pollutant explorer")
    available=[p for p in POLLUTANTS if p in df.columns]
    if not available: empty_chart("No recognized pollutant columns were found."); return
    p=st.selectbox("Choose pollutant",available,key="pollutant_main")
    l,r=st.columns(2)
    with l:
        vals=df[p].dropna()
        if len(vals):
            fig=px.histogram(vals,x=p,nbins=45,color_discrete_sequence=["#818cf8"],title=f"{p} concentration distribution")
            st.plotly_chart(styled(fig),use_container_width=True)
        else: empty_chart()
    with r:
        comp=df[available].median().dropna().sort_values(ascending=False).reset_index()
        comp.columns=["Pollutant","Median concentration"]
        if len(comp):
            fig=px.bar(comp,x="Pollutant",y="Median concentration",color="Pollutant",title="Median pollutant concentrations")
            st.plotly_chart(styled(fig),use_container_width=True)
        else: empty_chart()
    st.markdown("### Concentration over time")
    trend=df.groupby(pd.Grouper(key="Date",freq="MS"))[p].mean().dropna().reset_index()
    if len(trend):
        fig=px.line(trend,x="Date",y=p,markers=True,title=f"Monthly mean {p}")
        st.plotly_chart(styled(fig),use_container_width=True)
    else: empty_chart()
    if "AQI" in df:
        pair=df[[p,"AQI"]].dropna()
        if len(pair):
            fig=px.scatter(pair,x=p,y="AQI",opacity=.48,color_discrete_sequence=["#22d3ee"],
                           title=f"{p} vs AQI (exploratory association)")
            st.plotly_chart(styled(fig),use_container_width=True)
        else: empty_chart("No paired AQI and pollutant values.")
    st.markdown("### Pollutant correlation matrix")
    corr=df[available+([ "AQI"] if "AQI" in df else [])].corr(numeric_only=True)
    if corr.shape[0]>1:
        fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto",title="Pearson correlation")
        st.plotly_chart(styled(fig,520),use_container_width=True)
        pairs=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool)).stack().abs().sort_values(ascending=False)
        if len(pairs):
            a,b=pairs.index[0]
            st.info(f"Strongest absolute pairwise correlation in this selection: **{a} ↔ {b}** (|r| = {pairs.iloc[0]:.3f}). Correlation is not causation.")

def render_temporal(df):
    st.markdown("## Temporal analysis")
    if "AQI" not in df or not df.AQI.notna().any(): empty_chart("No AQI observations available for temporal analysis."); return
    daily=df.groupby("Date")["AQI"].mean().dropna().reset_index()
    monthly=df.groupby(pd.Grouper(key="Date",freq="MS"))["AQI"].mean().dropna().reset_index()
    c1,c2=st.columns(2)
    with c1:
        fig=px.line(daily,x="Date",y="AQI",title="Daily mean AQI across selected cities")
        st.plotly_chart(styled(fig,380),use_container_width=True)
    with c2:
        fig=px.line(monthly,x="Date",y="AQI",markers=True,title="Monthly mean AQI")
        st.plotly_chart(styled(fig,380),use_container_width=True)
    y=df.groupby("Year")["AQI"].mean().dropna().reset_index()
    if len(y):
        fig=px.bar(y,x="Year",y="AQI",color="AQI",color_continuous_scale=["#22d3ee","#a78bfa","#fb7185"],title="Year-wise average AQI")
        st.plotly_chart(styled(fig),use_container_width=True)
    if "Season" in df:
        seas=df.groupby("Season")["AQI"].mean().reindex(["Winter","Summer","Monsoon","Post-monsoon"]).dropna().reset_index()
        if len(seas):
            fig=px.bar(seas,x="Season",y="AQI",color="AQI",color_continuous_scale=["#22d3ee","#fb7185"],title="Seasonal average AQI")
            st.plotly_chart(styled(fig),use_container_width=True)
    pivot=df.pivot_table(index="Year",columns="Month",values="AQI",aggfunc="mean")
    if not pivot.empty:
        fig=px.imshow(pivot,text_auto=".0f",aspect="auto",color_continuous_scale="Viridis",title="Monthly AQI heatmap by year",
                      labels=dict(x="Month",y="Year",color="Mean AQI"))
        st.plotly_chart(styled(fig,360),use_container_width=True)

def render_risk(df):
    st.markdown("## AQI risk analysis")
    if "AQI_Bucket" not in df: empty_chart("AQI_Bucket column is not present."); return
    counts=df.AQI_Bucket.value_counts().reindex(AQI_ORDER).dropna()
    counts=counts[counts>0]
    if counts.empty: empty_chart("No labelled AQI category observations in the current selection."); return
    l,r=st.columns(2)
    with l:
        fig=px.pie(names=counts.index,values=counts.values,hole=.62,color_discrete_sequence=COLORS,title="AQI category share")
        fig.update_traces(textposition="inside",textinfo="percent+label")
        st.plotly_chart(styled(fig),use_container_width=True)
    with r:
        tab=(counts/counts.sum()*100).rename("Share (%)").to_frame()
        tab["Observations"]=counts
        st.dataframe(tab.style.format({"Share (%)":"{:.1f}%","Observations":"{:,.0f}"}),use_container_width=True)
        st.metric("High-risk share (Poor + Very Poor + Severe)",
                  f"{counts.reindex(['Poor','Very Poor','Severe']).fillna(0).sum()/counts.sum()*100:.1f}%")
    city=df.dropna(subset=["AQI_Bucket"]).assign(high_risk=lambda x:x.AQI_Bucket.isin(["Poor","Very Poor","Severe"]))
    risk=city.groupby("City").high_risk.mean().mul(100).sort_values(ascending=False).dropna().reset_index(name="High-risk share (%)")
    if len(risk):
        fig=px.bar(risk,x="City",y="High-risk share (%)",color="High-risk share (%)",color_continuous_scale=["#22d3ee","#fb7185"],title="City-wise high-risk category share")
        fig.update_layout(xaxis_tickangle=-40,coloraxis_showscale=False)
        st.plotly_chart(styled(fig,400),use_container_width=True)
        top=risk.iloc[0]
        st.info(f"Highest high-risk category share in this filtered view: **{top.City}** ({top['High-risk share (%)']:.1f}%).")

def render_advanced(df):
    st.markdown("## Advanced analytics")
    numeric=[c for c in POLLUTANTS+["AQI"] if c in df.columns and df[c].notna().any()]
    if not numeric: empty_chart("No numeric pollutant/AQI fields available."); return
    st.markdown("### Descriptive statistics")
    st.dataframe(df[numeric].describe().T.rename_axis("Feature"),use_container_width=True)
    st.markdown("### IQR outlier screening")
    rows=[]
    for c in numeric:
        s=df[c].dropna()
        if len(s):
            q1,q3=s.quantile([.25,.75]); iqr=q3-q1
            n=((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()
            rows.append({"Feature":c,"Q1":q1,"Q3":q3,"IQR":iqr,"Potential outliers":int(n),"Valid n":len(s)})
    out=pd.DataFrame(rows)
    st.dataframe(out,use_container_width=True)
    st.caption("IQR flags are screening indicators, not proof of measurement errors. Extreme observations are retained.")
    st.markdown("### PCA and K-Means pollutant profiles")
    chosen=[c for c in POLLUTANTS if c in df.columns and df[c].notna().any()]
    if len(chosen)<2: empty_chart("At least two pollutant columns are required for PCA and clustering."); return
    sample=df[chosen].copy()
    sample=sample.replace([np.inf,-np.inf],np.nan)
    sample=sample.dropna(how="all")
    maxn=12000
    if len(sample)>maxn: sample=sample.sample(maxn,random_state=42)
    if len(sample)<5: empty_chart("Too few pollutant observations for dimensionality reduction."); return
    prep=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    X=prep.fit_transform(sample)
    pca=PCA(n_components=2,random_state=42)
    coords=pca.fit_transform(X)
    pca_df=pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1]},index=sample.index)
    if "AQI_Bucket" in df:
        pca_df["AQI category"]=df.loc[sample.index,"AQI_Bucket"].fillna("Unlabelled").astype(str)
    l,r=st.columns(2)
    with l:
        fig=px.scatter(pca_df,x="PC1",y="PC2",color="AQI category" if "AQI category" in pca_df else None,
                       opacity=.65,title=f"PCA projection · variance retained {pca.explained_variance_ratio_.sum()*100:.1f}%")
        st.plotly_chart(styled(fig,420),use_container_width=True)
        st.caption(f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}% · PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
    with r:
        k=st.slider("K-Means cluster count",2,6,3,key="kmeans_k")
        km=KMeans(n_clusters=k,random_state=42,n_init=10)
        labels=km.fit_predict(X)
        pca_df["Cluster"]=labels.astype(str)
        fig=px.scatter(pca_df,x="PC1",y="PC2",color="Cluster",opacity=.65,title=f"K-Means profiles · k={k}")
        st.plotly_chart(styled(fig,420),use_container_width=True)
        sizes=pd.Series(labels).value_counts().sort_index().rename_axis("Cluster").to_frame("Observations")
        st.dataframe(sizes,use_container_width=True)
    centers=pd.DataFrame(prep.named_steps["scale"].inverse_transform(km.cluster_centers_),columns=chosen) if False else None
    # Interpret clusters using original-scale means from assigned rows
    profile=sample.assign(Cluster=labels).groupby("Cluster").mean(numeric_only=True)
    st.markdown("#### Cluster profile means (original measurement units)")
    st.dataframe(profile,use_container_width=True)
    st.caption("Clusters are algorithmic groupings, not validated AQI categories. PCA/K-Means use median imputation and standardization within this view.")

def next_observation_model(df):
    required=[p for p in POLLUTANTS if p in df.columns]
    if "AQI_Bucket" not in df or not required:
        return None,"Model needs AQI_Bucket and at least one pollutant feature."
    if "AQI_Bucket" not in df.columns:
        return None,"AQI_Bucket column is not present."
    work=df[["City","Date","AQI_Bucket"]+required].copy()
    work=work.sort_values(["City","Date"])
    # Target is the category on the immediately next available city observation.
    work["Target"]=work.groupby("City")["AQI_Bucket"].shift(-1)
    work["TargetDate"]=work.groupby("City")["Date"].shift(-1)
    # A transition is usable only when the next observation has a known category.
    work=work.dropna(subset=["Target","TargetDate"]).copy()
    if work["Target"].nunique()<2: return None,"At least two target classes are required."
    # Split by the date of the target observation, not by the feature-row date.
    work=work.sort_values(["TargetDate","City"])
    cut=int(len(work)*0.8)
    if cut<10 or len(work)-cut<5: return None,"Not enough labelled city-observation transitions for a chronological split."
    train,test=work.iloc[:cut],work.iloc[cut:]
    if train.Target.nunique()<2 or test.Target.nunique()<1:
        return None,"Chronological split does not contain enough class diversity to evaluate."
    cats=["City"]
    nums=required
    pre=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats)
    ])
    model=Pipeline([("preprocess",pre),("classifier",RandomForestClassifier(n_estimators=250,min_samples_leaf=2,
                        class_weight="balanced_subsample",random_state=42,n_jobs=-1))])
    Xtr=train[nums+cats]; Xte=test[nums+cats]
    model.fit(Xtr,train.Target)
    pred=model.predict(Xte)
    labels=sorted(set(test.Target)|set(pred))
    metrics={"Accuracy":accuracy_score(test.Target,pred),
             "Precision (weighted)":precision_score(test.Target,pred,average="weighted",zero_division=0),
             "Recall (weighted)":recall_score(test.Target,pred,average="weighted",zero_division=0),
             "F1 (weighted)":f1_score(test.Target,pred,average="weighted",zero_division=0)}
    return (model,metrics,test,pred,labels,nums,train),None

def render_ml(df):
    st.markdown("## Machine learning insights")
    st.markdown("**Experiment:** predict the AQI category at the next available labelled observation within the same city, using the current observation's pollutant readings and city. This is not necessarily next-day prediction.")
    st.warning("This is an educational historical experiment, not a live forecast or a public-health decision tool. The chronological holdout uses later target dates; missing pollutant values are imputed inside the training pipeline.")
    result,err=next_observation_model(df)
    if err: st.error(err); return
    model,metrics,test,pred,labels,features,train=result
    cols=st.columns(4)
    for c,(k,v) in zip(cols,metrics.items()):
        with c: metric_card(k,f"{v:.3f}","Chronological holdout")
    cm=confusion_matrix(test.Target,pred,labels=labels)
    l,r=st.columns([1,1])
    with l:
        fig=px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale="Blues",
                      labels=dict(x="Predicted",y="Actual",color="Count"),title="Confusion matrix")
        st.plotly_chart(styled(fig,430),use_container_width=True)
    with r:
        report=classification_report(test.Target,pred,labels=labels,output_dict=True,zero_division=0)
        st.dataframe(pd.DataFrame(report).T,use_container_width=True)
        st.caption(f"Training transitions: {len(train):,} · Test transitions: {len(test):,} · Test target dates: {test.TargetDate.min():%d %b %Y} to {test.TargetDate.max():%d %b %Y}")
    st.markdown("### Permutation feature importance")
    try:
        perm=permutation_importance(model,test[features+["City"]],test.Target,n_repeats=4,random_state=42,
                                    scoring="f1_weighted",n_jobs=-1)
        imp=pd.DataFrame({"Feature":features+["City"],"Importance":perm.importances_mean}).sort_values("Importance",ascending=False)
        fig=px.bar(imp,x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale=["#22d3ee","#a78bfa"])
        st.plotly_chart(styled(fig,400),use_container_width=True)
    except Exception as e:
        st.info(f"Feature importance could not be calculated for this split: {e}")
    st.markdown("#### Interpretation & limitations")
    st.markdown("- Target is the next available labelled record for the same city, not necessarily the following calendar day.")
    st.markdown("- The split is chronological by the date of the next observation being predicted; the model is trained on earlier target dates and evaluated on later target dates.")
    st.markdown("- Missing pollutant values are imputed using training-fold medians; city is one-hot encoded with unseen categories ignored.")
    st.markdown("- Dataset gaps, AQI class imbalance, historical coverage ending in 2020, and absent weather/traffic/station variables limit generalization.")
    st.markdown("- Scores are computed at runtime from the loaded data. They may differ from figures in the written report because this dashboard independently reconstructs the experiment.")

def render_explorer(df):
    st.markdown("## Data explorer")
    st.caption("Search, inspect missingness, and export the currently filtered rows.")
    q=st.text_input("Search rows (city, category or text)",placeholder="e.g. Delhi, Severe")
    view=df.copy()
    if q:
        mask=view.astype(str).apply(lambda col: col.str.contains(q,case=False,na=False)).any(axis=1)
        view=view[mask]
    available_pollutants=[p for p in POLLUTANTS if p in view.columns and view[p].notna().any()]
    if available_pollutants:
        chosen_pollutant=st.selectbox("Optional pollutant range filter",["No pollutant filter"]+available_pollutants,key="explorer_pollutant")
        if chosen_pollutant!="No pollutant filter":
            values=view[chosen_pollutant].dropna()
            lo,hi=float(values.min()),float(values.max())
            if lo < hi:
                selected_range=st.slider(f"{chosen_pollutant} range",min_value=lo,max_value=hi,value=(lo,hi),key="explorer_pollutant_range")
                view=view[view[chosen_pollutant].between(selected_range[0],selected_range[1])]
            else:
                st.caption(f"{chosen_pollutant} has a single observed value in this selection.")
    st.write(f"**{len(view):,} rows**")
    st.dataframe(view,use_container_width=True,height=440)
    csv=view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download filtered CSV",csv,file_name="airguard_filtered_data.csv",mime="text/csv")
    st.markdown("### Missing-value audit")
    miss=df.isna().sum().to_frame("Missing values")
    miss["Missing (%)"]=miss["Missing values"]/max(len(df),1)*100
    miss=miss.sort_values("Missing (%)",ascending=False)
    l,r=st.columns([1,1])
    with l: st.dataframe(miss.style.format({"Missing (%)":"{:.1f}%"}),use_container_width=True)
    with r:
        fig=px.bar(miss.reset_index(),x="Missing (%)",y="index",orientation="h",color="Missing (%)",
                   color_continuous_scale=["#22d3ee","#fb7185"],title="Missing values by column")
        fig.update_layout(yaxis_title="",coloraxis_showscale=False)
        st.plotly_chart(styled(fig,430),use_container_width=True)

def render_insights(df):
    st.markdown("## Insights & conclusion")
    if len(df)==0: empty_chart("No records match the current filters."); return
    bullets=[]
    if "AQI" in df and df.AQI.notna().any():
        means=df.groupby("City").AQI.mean().dropna()
        if len(means):
            bullets.append(f"Highest mean AQI city: **{means.idxmax()}** ({means.max():.1f}).")
            bullets.append(f"Lowest mean AQI city: **{means.idxmin()}** ({means.min():.1f}).")
        bymonth=df.groupby(df.Date.dt.to_period("M")).AQI.mean().dropna()
        if len(bymonth):
            bullets.append(f"Highest monthly mean AQI: **{bymonth.idxmax()}** ({bymonth.max():.1f}).")
        st.markdown("#### City and time signals")
    if "AQI_Bucket" in df and df.AQI_Bucket.notna().any():
        bucket=df.AQI_Bucket.value_counts()
        bullets.append(f"Most common AQI category: **{bucket.index[0]}** ({bucket.iloc[0]:,} records).")
    poll=[p for p in POLLUTANTS if p in df]
    if len(poll)>1 and "AQI" in df:
        corr=df[poll+["AQI"]].corr(numeric_only=True)["AQI"].drop("AQI").dropna()
        if len(corr):
            bullets.append(f"Strongest absolute pollutant–AQI correlation: **{corr.abs().idxmax()}** (r = {corr.loc[corr.abs().idxmax()]:.3f}). This is association, not causation.")
    for b in bullets: st.markdown("- "+b)
    if not bullets: st.info("No sufficiently complete fields are available to calculate insights.")
    st.markdown("### Interpretation")
    st.markdown("AIRGUARD INDIA summarizes historical air-quality observations from the selected period. City averages, pollutant correlations and AQI categories are descriptive; they do not establish causes. Missing observations and extreme values should be considered before drawing policy or health conclusions.")
    st.markdown("### Scope and data caveats")
    st.markdown("- Historical dataset: 2015–2020; not a live AQI monitoring system.")
    st.markdown("- AQI labels are missing for some records; supervised learning excludes transitions without valid target labels.")
    st.markdown("- Pollutant gaps are retained and handled per analysis; model imputation is fitted only on training data.")
    st.markdown("- IQR outliers are flagged, not automatically deleted. The unusually high maximum AQI should be investigated.")
    st.markdown("- The dataset does not include weather, traffic, station metadata or post-July-2020 observations.")

def main():
    st.markdown("""<div class="hero"><div class="eyebrow">AIR QUALITY INTELLIGENCE PLATFORM · INDIA</div>
    <div class="hero-title">AIRGUARD <span>INDIA</span></div>
    <div class="hero-sub">Intelligent Air Quality Analytics & AQI Risk Prediction</div>
    <div class="hero-foot"><span class="pulse"></span> HISTORICAL DATA MODE <span class="sep">/</span> 2015—2020 <span class="sep">/</span> DATA-DRIVEN INSIGHTS</div></div>""",unsafe_allow_html=True)
    uploaded=st.sidebar.file_uploader("Load city_day.csv or ZIP",type=["csv","zip"],help="Upload the Kaggle city_day.csv file or its ZIP archive.")
    source=uploaded if uploaded is not None else (DATA_PATH if DATA_PATH.exists() else None)
    if source is None:
        st.warning("Dataset not found. Upload city_day.csv or city_day.csv.zip using the sidebar, or place city_day.csv in the project's data/ folder.")
        st.markdown("Expected fields: City, Date, AQI_Bucket, AQI and available pollutant columns.")
        return
    try: raw=load_csv(source); df=normalize_data(raw)
    except Exception as e:
        st.error(f"Dataset validation failed: {e}"); return
    if df is None or df.empty:
        st.error("The uploaded dataset is empty or could not be parsed."); return
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ◈ Navigation")
    page=st.sidebar.radio("Go to",["Executive Overview","City Intelligence","Pollutant Explorer","Temporal Analysis","AQI Risk Analysis","Advanced Analytics","Machine Learning","Data Explorer","Insights & Conclusion"],label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Loaded · {len(df):,} rows × {len(df.columns)} columns")
    st.sidebar.caption(f"{df.City.nunique()} cities · {df.Date.min():%Y}–{df.Date.max():%Y}")
    st.sidebar.caption("Offline historical analytics · not live monitoring")
    filtered=filter_controls(df)
    st.markdown(f'<div class="view-label">CURRENT VIEW <b>{page.upper()}</b> <span>· {len(filtered):,} RECORDS</span></div>',unsafe_allow_html=True)
    if page=="Executive Overview": render_overview(filtered,df)
    elif page=="City Intelligence": render_city(filtered)
    elif page=="Pollutant Explorer": render_pollutants(filtered)
    elif page=="Temporal Analysis": render_temporal(filtered)
    elif page=="AQI Risk Analysis": render_risk(filtered)
    elif page=="Advanced Analytics": render_advanced(filtered)
    elif page=="Machine Learning": render_ml(filtered)
    elif page=="Data Explorer": render_explorer(filtered)
    elif page=="Insights & Conclusion": render_insights(filtered)
    st.markdown('<div class="footer">AIRGUARD INDIA · DATA ANALYTICS & VISUALIZATION · HISTORICAL DATA 2015–2020</div>',unsafe_allow_html=True)

if __name__ == "__main__":
    main()
