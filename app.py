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
    # Termos genéricos pré-carregados
    # ----------------------------
    default_generic_terms = [
        "male","female","age","human","adult","child","middle aged","in vitro",
        "in vivo","review","case report","animal","elderly","pregnancy",
        "female patient","male patient","newborn","adolescent","infant",
        "rat","mouse","study","clinical trial","controlled study"
    ]
    
    # Permitir usuário inserir/alterar termos genéricos
    generic_terms_selected = st.multiselect(
        "Termos genéricos (pré-carregados, você pode adicionar ou remover):",
        options=default_generic_terms,
        default=default_generic_terms
    )
    generic_terms = set([t.lower() for t in generic_terms_selected])
    
    # ----------------------------
    # Amostra de dados
    # ----------------------------
    sample = df_nonempty.head(10)[["Title","__date","__period_start","__terms_set"]]
    st.subheader("Amostra de registros processados")
    st.dataframe(sample)
    
    # ----------------------------
    # Construir série temporal por termo não genérico
    # ----------------------------
    all_terms = set()
    df_nonempty["__terms_set"].apply(lambda s: all_terms.update(s))
    non_generic_terms = all_terms - generic_terms
    st.write(f"Total de termos únicos (sem genéricos): {len(non_generic_terms)}")
    
    selected_terms = st.multiselect("Selecione termos para análise (não genéricos, ou vazio para todos)", list(non_generic_terms))
    if len(selected_terms) == 0:
        selected_terms = list(non_generic_terms)
    
    periods = sorted(df_nonempty["__period_start"].unique())
    
    # Termos não genéricos
    term_freq = {}
    for term in selected_terms:
        freq_list = []
        for p in periods:
            count = df_nonempty[df_nonempty["__period_start"]==p]["__terms_set"].apply(lambda s: term in s).sum()
            freq_list.append(count)
        term_freq[term] = freq_list
    
    df_freq = pd.DataFrame(term_freq, index=periods)
    st.subheader("Série temporal de termos não genéricos")
    st.dataframe(df_freq)
    
    # Termos genéricos
    generic_freq = {}
    for term in generic_terms:
        freq_list = []
        for p in periods:
            count = df_nonempty[df_nonempty["__period_start"]==p]["__terms_set"].apply(lambda s: term in s).sum()
            freq_list.append(count)
        generic_freq[term] = freq_list
    
    df_generic_freq = pd.DataFrame(generic_freq, index=periods)
    st.subheader("Série temporal de termos genéricos")
    st.dataframe(df_generic_freq)
    
    # ----------------------------
    # Cálculo métricas de tendência para termos não genéricos
    # ----------------------------
    slope_scores = {}
    z_scores = {}
    recency_scores = {}
    
    for term in selected_terms:
        y = np.log1p(df_freq[term].values)
        X = np.arange(len(y)).reshape(-1,1)
        
        # Regressão linear
        lr = LinearRegression().fit(X, y)
        slope = lr.coef_[0]
        slope_score = slope * (y[-1])
        slope_scores[term] = slope_score
        
        # Z-score de explosão
        if len(y) > 1:
            hist_mean = np.mean(y[:-1])
            hist_std = np.std(y[:-1])
            z = (y[-1]-hist_mean)/hist_std if hist_std>0 else 0
        else:
            z = 0
        z_scores[term] = z
        
        # Recency score
        window = min(12,len(y))
        freq_recent = np.sum(df_freq[term].values[-window:])
        freq_hist = np.sum(df_freq[term].values[:-window])
        recency = freq_recent / (freq_hist+1)
        recency_scores[term] = recency
    
    df_metrics = pd.DataFrame({
        "Termo": selected_terms,
        "SlopeScore": [slope_scores[t] for t in selected_terms],
        "Z-score": [z_scores[t] for t in selected_terms],
        "RecencyScore": [recency_scores[t] for t in selected_terms]
    }).sort_values("SlopeScore", ascending=False)
    
    st.subheader("Métricas de tendência por termo (não genéricos)")
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
