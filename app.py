from __future__ import annotations

import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEMO_FILE = APP_DIR / "data" / "donnees_pev_simulees.csv"
MONTHS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
VACCINES = {
    "BCG": "bcg",
    "Penta1": "penta1",
    "Penta3": "penta3",
    "VPI": "vpi",
    "Rougeole": "rougeole",
    "HPV": "hpv",
}
SOURCE_COLUMNS = {
    "annee": "year", "mois": "month", "province/prefecture": "region",
    "district sanitaire": "district", "population cible": "target",
    "bcg vaccines": "bcg", "penta1 vaccines": "penta1", "penta3 vaccines": "penta3",
    "vpi vaccines": "vpi", "rougeole vaccines": "rougeole", "hpv vaccines": "hpv",
    "rapports attendus": "reports_expected", "rapports recus": "reports_received",
    "rapports a temps": "reports_timely", "latitude": "latitude", "longitude": "longitude",
}


st.set_page_config(page_title="WHO Immunization Analytics", page_icon="💉", layout="wide")
st.markdown("""
<style>
:root{--navy:#17365d;--blue:#2563eb;--green:#15803d;--amber:#d97706;--red:#dc2626;--line:#dbe4ef}
.stApp{background:#f6f8fb}.block-container{max-width:1500px;padding-top:1rem}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
.hero{background:linear-gradient(120deg,#17365d,#176b73);padding:1.35rem 1.6rem;border-radius:18px;color:#fff;margin-bottom:.9rem}
.hero h1{color:#fff;font-size:2rem;margin:.15rem 0}.hero p{color:#dcecf2;margin:0}.hero small{font-weight:700;letter-spacing:.12em}
.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:.8rem 0 1rem}
.card{background:#fff;border:1px solid var(--line);border-top:4px solid var(--accent);border-radius:14px;padding:.85rem;box-shadow:0 4px 12px rgba(15,23,42,.05)}
.label{color:#64748b;font-size:.78rem;font-weight:650;min-height:2.1em}.value{font-size:1.55rem;font-weight:750;color:#172033}.hint{font-size:.72rem;color:var(--accent);font-weight:650}
div[data-testid="stPlotlyChart"],div[data-testid="stDataFrame"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.15rem;overflow:hidden}
.stDownloadButton button,.stButton button{border-radius:10px;min-height:2.6rem;font-weight:650}
@media(max-width:900px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.hero h1{font-size:1.5rem}}
@media(max-width:520px){.block-container{padding:.6rem}.cards{grid-template-columns:1fr 1fr}.value{font-size:1.25rem}}
</style>
""", unsafe_allow_html=True)


def norm(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value).strip().lower())
    return "".join(c for c in value if not unicodedata.combining(c))


def read_file(content: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content), sheet_name="Données mensuelles", header=3)
    for separator in (";", ",", "\t"):
        try:
            frame = pd.read_csv(io.BytesIO(content), sep=separator, encoding="utf-8-sig")
            if len(frame.columns) > 5:
                return frame
        except Exception:
            continue
    raise ValueError("Le fichier n’a pas pu être lu.")


@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    return pd.read_csv(DEMO_FILE, encoding="utf-8-sig")


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    rename = {col: SOURCE_COLUMNS.get(norm(col), norm(col).replace(" ", "_")) for col in frame.columns}
    frame = frame.rename(columns=rename)
    required = ["year", "month", "region", "district", "target", *VACCINES.values(), "reports_expected", "reports_received", "reports_timely", "latitude", "longitude"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError("Colonnes absentes : " + ", ".join(missing))
    numeric = ["year", "target", *VACCINES.values(), "reports_expected", "reports_received", "reports_timely", "latitude", "longitude"]
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=["year", "month", "district", "target"]).copy()
    frame["year"] = frame["year"].astype(int)
    frame["month_num"] = frame["month"].map({name: i + 1 for i, name in enumerate(MONTHS)})
    frame["period"] = pd.to_datetime(dict(year=frame["year"], month=frame["month_num"], day=1), errors="coerce")
    denominator = frame["target"].replace(0, np.nan)
    for label, name in VACCINES.items():
        frame[f"cov_{name}"] = frame[name] / denominator
    frame["dropout"] = (frame["penta1"] - frame["penta3"]) / frame["penta1"].replace(0, np.nan)
    frame["zero_dose"] = (frame["target"] - frame["penta1"]).clip(lower=0)
    frame["zero_rate"] = frame["zero_dose"] / denominator
    frame["completeness"] = frame["reports_received"] / frame["reports_expected"].replace(0, np.nan)
    frame["timeliness"] = frame["reports_timely"] / frame["reports_expected"].replace(0, np.nan)
    coverage_cols = [f"cov_{x}" for x in VACCINES.values()]
    frame["flag_coverage"] = frame[coverage_cols].gt(1).any(axis=1)
    frame["flag_penta"] = frame["penta3"] > frame["penta1"]
    frame["flag_reporting"] = (frame["reports_received"] > frame["reports_expected"]) | (frame["reports_timely"] > frame["reports_received"])
    frame["flag_missing"] = frame[required].isna().any(axis=1)
    frame["flag_quality"] = (frame["completeness"] < .95) | (frame["timeliness"] < .95)
    frame["dqa_issue"] = frame[["flag_coverage", "flag_penta", "flag_reporting", "flag_missing", "flag_quality"]].any(axis=1)
    coverage_deficit = (1 - frame["cov_penta3"] / .90).clip(0, 1)
    zero_component = (frame["zero_rate"] / .30).clip(0, 1)
    dropout_component = (frame["dropout"].clip(lower=0) / .30).clip(0, 1)
    quality_component = (1 - frame[["completeness", "timeliness"]].mean(axis=1)).clip(0, 1)
    frame["priority_score"] = 100 * (.40 * coverage_deficit + .30 * zero_component + .20 * dropout_component + .10 * quality_component)
    return frame


def aggregate(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    columns = ["target", *VACCINES.values(), "reports_expected", "reports_received", "reports_timely"]
    out = frame.groupby(group, as_index=False)[columns].sum()
    denominator = out["target"].replace(0, np.nan)
    for label, name in VACCINES.items():
        out[f"cov_{name}"] = out[name] / denominator
    out["dropout"] = (out["penta1"] - out["penta3"]) / out["penta1"].replace(0, np.nan)
    out["zero_dose"] = (out["target"] - out["penta1"]).clip(lower=0)
    out["zero_rate"] = out["zero_dose"] / denominator
    out["completeness"] = out["reports_received"] / out["reports_expected"].replace(0, np.nan)
    out["timeliness"] = out["reports_timely"] / out["reports_expected"].replace(0, np.nan)
    coverage_deficit = (1 - out["cov_penta3"] / .90).clip(0, 1)
    out["priority_score"] = 100 * (.40 * coverage_deficit + .30 * (out["zero_rate"] / .30).clip(0, 1) + .20 * (out["dropout"].clip(lower=0) / .30).clip(0, 1) + .10 * (1 - out[["completeness", "timeliness"]].mean(axis=1)).clip(0, 1))
    return out


def percent(value: float) -> str:
    return "N/D" if pd.isna(value) else f"{value:.1%}".replace(".", ",")


def integer(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def plot_style(fig, height=390):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=15, r=15, t=55, b=25), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial", color="#334155"), legend_title_text="")
    fig.update_xaxes(gridcolor="#edf2f7"); fig.update_yaxes(gridcolor="#edf2f7")
    return fig


PLOT_CONFIG = {"displaylogo": False, "responsive": True, "toImageButtonOptions": {"format": "png", "filename": "graphique_pev", "scale": 2}}

st.markdown('<section class="hero"><small>CAS D’ÉTUDE PEV</small><h1>WHO Immunization Analytics</h1><p>Couverture vaccinale, abandon, enfants zéro dose, qualité des données et priorisation géographique.</p></section>', unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("Importer les données PEV", type=["xlsx", "xls", "csv"])
try:
    raw = read_file(uploaded.getvalue(), uploaded.name) if uploaded else load_demo()
    df = prepare(raw)
except Exception as exc:
    st.error(f"Chargement impossible : {exc}"); st.stop()

with st.sidebar:
    st.caption("Source : " + (uploaded.name if uploaded else "données simulées 2024–2026"))
    st.subheader("Filtres")
    years = sorted(df["year"].unique())
    selected_years = st.multiselect("Année", years, default=years)
    if selected_years: df = df[df["year"].isin(selected_years)]
    regions = sorted(df["region"].dropna().astype(str).unique())
    selected_regions = st.multiselect("Province/Préfecture", regions, default=regions)
    if selected_regions: df = df[df["region"].astype(str).isin(selected_regions)]
    districts = sorted(df["district"].dropna().astype(str).unique())
    selected_districts = st.multiselect("District", districts, default=districts)
    if selected_districts: df = df[df["district"].astype(str).isin(selected_districts)]

if df.empty:
    st.warning("Aucune donnée ne correspond aux filtres."); st.stop()

totals = aggregate(df, ["year"]).sum(numeric_only=True)
target = df["target"].sum(); p1 = df["penta1"].sum(); p3 = df["penta3"].sum()
cov_p3 = p3 / target if target else np.nan
dropout = (p1 - p3) / p1 if p1 else np.nan
zero = max(0, target - p1); zero_rate = zero / target if target else np.nan
completeness = df["reports_received"].sum() / df["reports_expected"].sum() if df["reports_expected"].sum() else np.nan
st.markdown(f"""<div class="cards">
<div class="card" style="--accent:#2563eb"><div class="label">Couverture Penta3</div><div class="value">{percent(cov_p3)}</div><div class="hint">objectif indicatif 90 %</div></div>
<div class="card" style="--accent:#d97706"><div class="label">Abandon Penta1–Penta3</div><div class="value">{percent(dropout)}</div><div class="hint">seuil indicatif 10 %</div></div>
<div class="card" style="--accent:#7c3aed"><div class="label">Enfants zéro dose</div><div class="value">{integer(zero)}</div><div class="hint">{percent(zero_rate)} de la cible</div></div>
<div class="card" style="--accent:#15803d"><div class="label">Complétude</div><div class="value">{percent(completeness)}</div><div class="hint">rapports reçus / attendus</div></div>
<div class="card" style="--accent:#dc2626"><div class="label">Lignes avec alerte DQA</div><div class="value">{int(df['dqa_issue'].sum())}</div><div class="hint">sur {len(df)} observations</div></div></div>""", unsafe_allow_html=True)

module = st.radio("Module", ["Vue nationale", "Couverture", "Abandon", "Zéro dose", "Qualité DQA", "Géospatial", "Assistant analytique"], horizontal=True, label_visibility="collapsed")
st.caption("Les seuils sont paramétrables et doivent être validés selon les normes nationales. Les graphiques peuvent être téléchargés avec l’icône appareil photo.")

if module == "Vue nationale":
    annual = aggregate(df, ["year"])
    chart_data = annual[["year", "cov_bcg", "cov_penta1", "cov_penta3", "cov_vpi", "cov_rougeole", "cov_hpv"]].rename(columns={"cov_bcg":"BCG", "cov_penta1":"Penta1", "cov_penta3":"Penta3", "cov_vpi":"VPI", "cov_rougeole":"Rougeole", "cov_hpv":"HPV"}).melt("year", var_name="Vaccin", value_name="Couverture")
    fig = px.line(chart_data, x="year", y="Couverture", color="Vaccin", markers=True, title="Tendances sur trois ans")
    fig.update_yaxes(tickformat=".0%", range=[0, max(1.05, chart_data["Couverture"].max() * 1.08)])
    st.plotly_chart(plot_style(fig, 430), width="stretch", config=PLOT_CONFIG)
    st.download_button("Télécharger la synthèse annuelle", csv_bytes(annual), "synthese_annuelle_pev.csv", "text/csv")

elif module == "Couverture":
    vaccine_label = st.selectbox("Vaccin", list(VACCINES))
    vaccine = VACCINES[vaccine_label]
    monthly = aggregate(df, ["period"])
    fig = px.line(monthly, x="period", y=f"cov_{vaccine}", markers=True, title=f"Évolution mensuelle de la couverture {vaccine_label}")
    fig.add_hline(y=.90, line_dash="dash", line_color="#15803d", annotation_text="Objectif 90 %")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(plot_style(fig), width="stretch", config=PLOT_CONFIG)
    ranked = aggregate(df, ["district"])[["district", f"cov_{vaccine}"]].sort_values(f"cov_{vaccine}")
    c1, c2 = st.columns(2)
    bottom = ranked.head(10); top = ranked.tail(10).sort_values(f"cov_{vaccine}")
    c1.plotly_chart(plot_style(px.bar(bottom, x=f"cov_{vaccine}", y="district", orientation="h", title="Bottom 10 districts", color_discrete_sequence=["#dc2626"])), width="stretch", config=PLOT_CONFIG)
    c2.plotly_chart(plot_style(px.bar(top, x=f"cov_{vaccine}", y="district", orientation="h", title="Top 10 districts", color_discrete_sequence=["#15803d"])), width="stretch", config=PLOT_CONFIG)
    st.download_button("Télécharger le classement", csv_bytes(ranked), f"classement_{vaccine}.csv", "text/csv")

elif module == "Abandon":
    district = aggregate(df, ["district"])[["district", "dropout"]].sort_values("dropout", ascending=False)
    fig = px.bar(district, x="dropout", y="district", orientation="h", title="Classement des districts", color="dropout", color_continuous_scale=["#15803d", "#f59e0b", "#dc2626"])
    fig.update_xaxes(tickformat=".0%"); fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(plot_style(fig, 480), width="stretch", config=PLOT_CONFIG)
    quarter = df.assign(trimestre=df["period"].dt.to_period("Q").astype(str)).pipe(aggregate, ["trimestre"])
    qfig = px.line(quarter, x="trimestre", y="dropout", markers=True, title="Évolution trimestrielle de l’abandon")
    qfig.add_hline(y=.10, line_dash="dash", line_color="#dc2626", annotation_text="Seuil 10 %"); qfig.update_yaxes(tickformat=".0%")
    st.plotly_chart(plot_style(qfig), width="stretch", config=PLOT_CONFIG)

elif module == "Zéro dose":
    district = aggregate(df, ["district"])[["district", "zero_dose", "zero_rate"]].sort_values("zero_dose", ascending=False)
    c1, c2 = st.columns([1.2, 1])
    fig = px.bar(district.head(15).sort_values("zero_dose"), x="zero_dose", y="district", orientation="h", title="Distribution des enfants zéro dose", color="zero_rate", color_continuous_scale=["#fde68a", "#dc2626"])
    fig.update_layout(coloraxis_colorbar_title="Taux")
    c1.plotly_chart(plot_style(fig, 500), width="stretch", config=PLOT_CONFIG)
    c2.dataframe(district.rename(columns={"district":"District", "zero_dose":"Zéro dose", "zero_rate":"Taux zéro dose"}), width="stretch", hide_index=True, column_config={"Taux zéro dose": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=.30)})
    st.download_button("Télécharger les données zéro dose", csv_bytes(district), "zero_dose_par_district.csv", "text/csv")

elif module == "Qualité DQA":
    quality = pd.DataFrame({"Contrôle": ["Couverture > 100 %", "Penta3 > Penta1", "Rapportage incohérent", "Valeurs manquantes", "Complétude ou promptitude faible"], "Alertes": [df["flag_coverage"].sum(), df["flag_penta"].sum(), df["flag_reporting"].sum(), df["flag_missing"].sum(), df["flag_quality"].sum()]})
    fig = px.bar(quality, x="Alertes", y="Contrôle", orientation="h", title="Alertes de qualité des données", color="Alertes", color_continuous_scale=["#fef3c7", "#dc2626"])
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(plot_style(fig), width="stretch", config=PLOT_CONFIG)
    anomalies = df[df["dqa_issue"]].copy()
    shown = anomalies[["year", "month", "region", "district", "flag_coverage", "flag_penta", "flag_reporting", "flag_missing", "flag_quality"]]
    st.dataframe(shown, width="stretch", hide_index=True)
    st.download_button("Télécharger les anomalies DQA", csv_bytes(shown), "anomalies_dqa.csv", "text/csv")

elif module == "Géospatial":
    geo = aggregate(df, ["district"])
    coords = df.groupby("district", as_index=False)[["latitude", "longitude"]].mean()
    geo = geo.merge(coords, on="district", how="left")
    geo["niveau"] = pd.cut(geo["priority_score"], bins=[-1,25,50,75,101], labels=["Faible", "Modéré", "Élevé", "Critique"])
    fig = px.scatter_map(geo, lat="latitude", lon="longitude", size="zero_dose", color="niveau", hover_name="district", hover_data={"priority_score":":.1f", "cov_penta3":":.1%", "dropout":":.1%", "zero_rate":":.1%"}, color_discrete_map={"Faible":"#15803d", "Modéré":"#eab308", "Élevé":"#f97316", "Critique":"#dc2626"}, zoom=4.2, height=570, title="Carte décisionnelle des districts prioritaires", map_style="open-street-map")
    fig.update_layout(margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    ranking = geo.sort_values("priority_score", ascending=False)
    st.dataframe(ranking[["district", "priority_score", "cov_penta3", "dropout", "zero_dose", "completeness", "niveau"]], width="stretch", hide_index=True)
    st.download_button("Télécharger le classement prioritaire", csv_bytes(ranking), "districts_prioritaires.csv", "text/csv")

else:
    district_table = aggregate(df, ["district"]).sort_values("priority_score", ascending=False)
    selected = st.selectbox("District à expliquer", district_table["district"].tolist())
    row = district_table[district_table["district"] == selected].iloc[0]
    level = "critique" if row["priority_score"] >= 75 else "élevée" if row["priority_score"] >= 50 else "modérée" if row["priority_score"] >= 25 else "faible"
    st.subheader(f"Pourquoi {selected} est-il prioritaire ?")
    st.info(f"Le district présente une couverture Penta3 de {percent(row['cov_penta3'])}, un taux d’abandon de {percent(row['dropout'])}, environ {integer(row['zero_dose'])} enfants zéro dose et une complétude de {percent(row['completeness'])}. Son score de priorité est de {row['priority_score']:.1f}/100, correspondant à une priorité {level}.")
    st.caption("Cette explication est générée à partir des indicateurs calculés. Elle ne remplace pas l’interprétation d’un responsable du programme.")

st.divider()
st.download_button("Télécharger toutes les données filtrées", csv_bytes(df), "donnees_pev_filtrees.csv", "text/csv", width="stretch")
