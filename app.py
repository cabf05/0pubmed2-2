import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
        else:
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
    # Termos genéricos
    # ----------------------------
    default_generic_terms = [
        "male","female","age","human","adult","child","middle aged","in vitro",
        "in vivo","review","case report","animal","elderly","pregnancy","newborn",
        "adolescent","infant","rat","mouse","study","clinical trial","controlled study",
        "female patient","male patient"
    ]
    st.subheader("Termos genéricos")
    generic_terms_input = st.text_area(
        "Edite ou adicione termos genéricos separados por vírgula:",
        value=", ".join(default_generic_terms),
        height=100
    )
    generic_terms = set([t.strip().lower() for t in generic_terms_input.split(",") if t.strip() != ""])

    # ----------------------------
    # Mostrar amostra
    # ----------------------------
    sample = df_nonempty.head(10)[["Title","__date","__period_start","__terms_set"]]
    st.subheader("Amostra de registros processados")
    st.dataframe(sample)

    # ----------------------------
    # Construir série temporal por termo
    # ----------------------------
    all_terms = set()
    df_nonempty["__terms_set"].apply(lambda s: all_terms.update(s))
    all_terms_no_generic = all_terms - generic_terms
    st.write(f"Total de termos únicos (sem genéricos): {len(all_terms_no_generic)}")

    selected_terms = st.multiselect(
        "Selecione termos para análise (ou deixe vazio para analisar todos)", 
        list(all_terms_no_generic)
    )
    if len(selected_terms) == 0:
        selected_terms = list(all_terms_no_generic)

    periods = sorted(df_nonempty["__period_start"].unique())
    term_freq = {}
    for term in selected_terms:
        freq_list = []
        for p in periods:
            count = df_nonempty[df_nonempty["__period_start"]==p]["__terms_set"].apply(lambda s: term in s).sum()
            freq_list.append(count)
        term_freq[term] = freq_list
    df_freq = pd.DataFrame(term_freq, index=periods)
    st.subheader("Série temporal de frequência por termo")
    st.dataframe(df_freq)

    # ----------------------------
    # Tabela termos genéricos
    # ----------------------------
    generic_freq = {}
    for term in generic_terms:
        freq_list = []
        for p in periods:
            count = df_nonempty[df_nonempty["__period_start"]==p]["__terms_set"].apply(lambda s: term in s).sum()
            freq_list.append(count)
        generic_freq[term] = freq_list
    df_generic_freq = pd.DataFrame(generic_freq, index=periods)
    st.subheader("Frequência de termos genéricos por período")
    st.dataframe(df_generic_freq)

    # ----------------------------
    # Métricas de tendência
    # ----------------------------
    slope_scores, z_scores, recency_scores = {}, {}, {}
    for term in selected_terms:
        y = np.log1p(df_freq[term].values)
        X = np.arange(len(y)).reshape(-1,1)
        lr = LinearRegression().fit(X, y)
        slope = lr.coef_[0]
        slope_scores[term] = slope * y[-1]
        if len(y) > 1:
            hist_mean = np.mean(y[:-1])
            hist_std = np.std(y[:-1])
            z = (y[-1]-hist_mean)/hist_std if hist_std>0 else 0
        else:
            z = 0
        z_scores[term] = z
        window = min(12,len(y))
        freq_recent = np.sum(df_freq[term].values[-window:])
        freq_hist = np.sum(df_freq[term].values[:-window])
        recency_scores[term] = freq_recent / (freq_hist+1)

    df_metrics = pd.DataFrame({
        "Termo": selected_terms,
        "SlopeScore": [slope_scores[t] for t in selected_terms],
        "Z-score": [z_scores[t] for t in selected_terms],
        "RecencyScore": [recency_scores[t] for t in selected_terms],
        "Search Volume": [np.sum(df_freq[t].values) for t in selected_terms],
        "Started": [ "Yes" if recency_scores[t] > 1.5 else "No" for t in selected_terms],
        "Trend Breakdown": [", ".join([c for c in term_cols if any(df_nonempty[df_nonempty[c].notna()][c].str.lower().str.contains(t))]) for t in selected_terms]
    }).sort_values("SlopeScore", ascending=False)
    st.subheader("Tendências por termo (estilo Google Trends)")
    st.dataframe(df_metrics)

    # ----------------------------
    # Gráfico estilo Google Trends
    # ----------------------------
    st.subheader("Visualização de tendência")
    selected_for_plot = st.multiselect("Selecione termos para visualizar no gráfico", selected_terms, default=selected_terms[:5])
    if selected_for_plot:
        plot_data = df_freq[selected_for_plot]
        st.line_chart(plot_data)

    # ----------------------------
    # Nuvem de palavras
    # ----------------------------
    st.subheader("Nuvem de palavras (sem termos genéricos)")
    all_text = " ".join([" ".join([t for t in s if t not in generic_terms]) for s in df_nonempty["__terms_set"]])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
