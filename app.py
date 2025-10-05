import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Tendências PubMed", layout="wide")
st.title("Análise de Tendências de Termos em Artigos PubMed")

# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Carregue o arquivo CSV do PubMed", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Arquivo carregado com {len(df)} registros.")

    # ----------------------------
    # Escolha de colunas de data
    # ----------------------------
    col_pub = "Date"
    col_create = "Date - Create"
    df["__date"] = pd.to_datetime(df[col_pub], errors='coerce')
    df["__date"] = df["__date"].fillna(pd.to_datetime(df[col_create], errors='coerce'))
    df_nonempty = df.dropna(subset=["__date"]).copy()
    st.write(f"Registros com data válida: {len(df_nonempty)}")

    # ----------------------------
    # Escolha granularidade
    # ----------------------------
    interval_choice = st.selectbox("Escolha granularidade da série temporal", ["Mensal", "Trimestral", "Anual"])
    def normalize_period(date):
        if interval_choice == "Mensal":
            return pd.Timestamp(date.year, date.month, 1)
        elif interval_choice == "Trimestral":
            quarter = (date.month-1)//3*3 + 1
            return pd.Timestamp(date.year, quarter, 1)
        else:  # Anual
            return pd.Timestamp(date.year, 1, 1)
    df_nonempty["__period_start"] = df_nonempty["__date"].apply(normalize_period)

    # ----------------------------
    # Preparar termos
    # ----------------------------
    term_cols = ["MeSH Terms", "Other Term", "Author Keywords"]
    def extract_terms(row):
        terms = set()
        for col in term_cols:
            if col in row and pd.notna(row[col]):
                for t in str(row[col]).split(";"):
                    t_clean = t.strip().lower()
                    if t_clean != "":
                        terms.add(t_clean)
        return terms
    df_nonempty["__terms_set"] = df_nonempty.apply(extract_terms, axis=1)

    # ----------------------------
    # Termos genéricos (pré-carregados)
    # ----------------------------
    default_generic_terms = [
        "male","female","age","human","adult","child","middle aged","in vitro",
        "in vivo","review","case report","animal","elderly","pregnancy","female patient",
        "male patient","newborn","adolescent","infant","rat","mouse","study",
        "clinical trial","controlled study"
    ]
    st.subheader("Configuração de termos genéricos")
    generic_terms_input = st.text_area(
        "Edite ou adicione novos termos genéricos (separados por vírgula):",
        value=", ".join(default_generic_terms),
        height=150
    )
    generic_terms = set([t.strip().lower() for t in generic_terms_input.split(",") if t.strip() != ""])

    # ----------------------------
    # Amostra de dados
    # ----------------------------
    sample = df_nonempty.head(10)[["Title","__date","__period_start","__terms_set"]]
    st.subheader("Amostra de registros processados")
    st.dataframe(sample)

    # ----------------------------
    # Construir série temporal por termo
    # ----------------------------
    all_terms = set()
    df_nonempty["__terms_set"].apply(lambda s: all_terms.update(s))
    all_terms_no_generic = sorted(all_terms - generic_terms)

    selected_terms = st.multiselect(
        "Selecione termos para análise (ou deixe vazio para analisar todos)", 
        all_terms_no_generic
    )
    if len(selected_terms) == 0:
        selected_terms = all_terms_no_generic

    # Matrizes de frequência
    periods = sorted(df_nonempty["__period_start"].unique())
    term_freq = {}
    generic_freq = {}
    for term in selected_terms:
        freq_list = []
        for p in periods:
            count = df_nonempty[df_nonempty["__period_start"]==p]["__terms_set"].apply(lambda s: term in s).sum()
            freq_list.append(count)
        term_freq[term] = freq_list

    for term in generic_terms:
        freq_list = []
        for p in periods:
            count = df_nonempty[df_nonempty["__period_start"]==p]["__terms_set"].apply(lambda s: term in s).sum()
            freq_list.append(count)
        generic_freq[term] = freq_list

    df_freq = pd.DataFrame(term_freq, index=periods)
    df_generic_freq = pd.DataFrame(generic_freq, index=periods)

    st.subheader("Série temporal de frequência por termo")
    st.dataframe(df_freq)

    st.subheader("Série temporal de termos genéricos")
    st.dataframe(df_generic_freq)

    # ----------------------------
    # Cálculo métricas de tendência
    # ----------------------------
    slope_scores = {}
    z_scores = {}
    recency_scores = {}
    for term in selected_terms:
        y = np.log1p(df_freq[term].values)
        X = np.arange(len(y)).reshape(-1,1)
        lr = LinearRegression().fit(X, y)
        slope = lr.coef_[0]
        slope_scores[term] = slope * y[-1]  # multiplicar por freq recente

        if len(y) > 1:
            hist_mean = np.mean(y[:-1])
            hist_std = np.std(y[:-1])
            z_scores[term] = (y[-1]-hist_mean)/hist_std if hist_std>0 else 0
        else:
            z_scores[term] = 0

        window = min(12,len(y))
        freq_recent = np.sum(df_freq[term].values[-window:])
        freq_hist = np.sum(df_freq[term].values[:-window])
        recency_scores[term] = freq_recent / (freq_hist+1)

    df_metrics = pd.DataFrame({
        "Termo": selected_terms,
        "SlopeScore": [slope_scores[t] for t in selected_terms],
        "Z-score": [z_scores[t] for t in selected_terms],
        "RecencyScore": [recency_scores[t] for t in selected_terms]
    }).sort_values("SlopeScore", ascending=False)
    st.subheader("Métricas de tendência por termo")
    st.dataframe(df_metrics)

    # ----------------------------
    # Nuvem de palavras
    # ----------------------------
    st.subheader("Nuvem de palavras")
    all_text = " ".join([" ".join(list(s)) for s in df_nonempty["__terms_set"]])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # ----------------------------
    # Visualização estilo Google Trends
    # ----------------------------
    st.subheader("Visualização estilo Google Trends")
    trend_breakdown = []
    for t in selected_terms:
        breakdown = []
        for col in term_cols:
            if col in df_nonempty.columns:
                col_values = df_nonempty[df_nonempty[col].notna()][col].astype(str).str.lower()
                if any(col_values.str.contains(t)):
                    breakdown.append(col)
        trend_breakdown.append(", ".join(breakdown))

    views_history_list = []
    for t in selected_terms:
        views_history_list.append([int(v) for v in df_freq[t].values.tolist()])

    trend_vis_df = pd.DataFrame({
        "Termo": selected_terms,
        "Search Volume": [int(np.sum(df_freq[t].values)) for t in selected_terms],
        "Started": ["Yes" if recency_scores[t]>1.5 else "No" for t in selected_terms],
        "Trend Breakdown": trend_breakdown,
        "Views History": views_history_list
    })

    st.dataframe(
        trend_vis_df,
        column_config={
            "Termo": "Termo",
            "Search Volume": st.column_config.NumberColumn("Search Volume"),
            "Started": "Trend Started",
            "Trend Breakdown": "Trend Breakdown",
            "Views History": st.column_config.LineChartColumn(
                "Artigos ao longo do tempo",
                y_min=0,
                y_max=max([max(v) for v in views_history_list])
            )
        },
        hide_index=True
    )
