# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re, unicodedata
from collections import Counter, defaultdict
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime

st.set_page_config(page_title="Term Trend Detector", layout="wide")

# ----------------- Utilitários -----------------
def clean_term(t):
    if not t or pd.isna(t): 
        return None
    s = str(t).strip()
    s = re.sub(r'^[\'"\s]+|[\'"\s]+$', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('ascii')
    s = s.lower().strip()
    return s if s else None

def split_cell(cell):
    if not cell or pd.isna(cell): 
        return []
    parts = re.split(r';|\||\n|\t', str(cell))
    out = []
    for p in parts:
        cp = clean_term(p)
        if cp:
            out.append(cp)
    return out

def choose_date(row, pub_col='Date - Publication', create_col='Date - Create'):
    # Try publication date first
    for c in (pub_col, create_col):
        v = row.get(c)
        if pd.isna(v) or v is None or str(v).strip() == '' or str(v).strip().upper() == 'N/A':
            continue
        # Try parse
        try:
            dt = pd.to_datetime(v, errors='coerce')
            if not pd.isna(dt):
                return dt
        except:
            continue
    return pd.NaT

def start_of_period(dt, granularity):
    if pd.isna(dt):
        return pd.NaT
    if granularity == 'Monthly':
        return pd.Timestamp(dt.year, dt.month, 1)
    if granularity == 'Quarterly':
        q = (dt.month - 1)//3 + 1
        month = 3*(q-1) + 1
        return pd.Timestamp(dt.year, month, 1)
    if granularity == 'Yearly':
        return pd.Timestamp(dt.year, 1, 1)
    return pd.Timestamp(dt.year, dt.month, 1)

def download_dataframe_as_csv(df):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# ----------------- UI -----------------
st.title("🔎 Term Trend Detector — Streamlit")
st.markdown("Carregue um CSV exportado do PubMed. O app unirá termos de `MeSH Terms` e `Author Keywords`, construirá séries temporais por termo (document frequency) e calculará 3 métricas de tendência.")

with st.expander("📄 Instruções / Observações"):
    st.write("""
    - O app usa `Date - Publication` quando disponível; senão `Date - Create`.  
    - Termos são separados por `;`, `|` ou quebra de linha. Não split por espaço (preserva frases).  
    - Métricas calculadas: Regressão linear (slope × freq recente), Z-score (último período), RecencyScore (janela recente).  
    - O usuário pode inserir termos genéricos para ignorar; uma lista inicial de ~30 já vem carregada.
    """)

# upload
uploaded = st.file_uploader("1) Faça upload do CSV (export PubMed)", type=["csv","txt"], accept_multiple_files=False)
cols_term_defaults = ["MeSH Terms", "Author Keywords"]
col1, col2 = st.columns(2)
with col1:
    gran = st.selectbox("2) Granularidade da série temporal", ['Monthly','Quarterly','Yearly'], index=0)
with col2:
    top_n = st.number_input("Top N por métrica", min_value=5, max_value=200, value=20, step=5)

# preloaded generic terms to ignore
preloaded_generic = [
    "male","female","human","humans","adult","child","children","aged","age","study","review",
    "case report","clinical trial","article","animals","in vitro","in vivo","trial","randomized",
    "observational study","cohort study","cross-sectional study","pregnancy","pregnant","female",
    "male","subject"
]
with st.expander("Termos genéricos (stop-terms) — editar se desejar"):
    user_generic = st.text_area("Adicione termos a excluir (separados por vírgula). Já carregados:", value=", ".join(preloaded_generic), height=140)
    custom_stop_terms = [clean_term(x) for x in re.split(r',|\n', user_generic) if clean_term(x)]
    st.write(f"{len(custom_stop_terms)} termos genéricos configurados.")

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded, dtype=str)
    except Exception as e:
        st.error(f"Erro ao ler CSV: {e}")
        st.stop()

    st.success(f"Arquivo carregado: {uploaded.name} — {df_raw.shape[0]} linhas, {df_raw.shape[1]} colunas")
    st.write("Colunas detectadas:", list(df_raw.columns))

    # allow user to choose columns if names differ
    st.markdown("### Opcional: ajuste os nomes das colunas se forem diferentes")
    col_pub = st.text_input("Coluna Data de Publicação (preferida)", value="Date - Publication")
    col_create = st.text_input("Coluna Date - Create (fallback)", value="Date - Create")
    col_mesh = st.text_input("Coluna MeSH Terms", value="MeSH Terms")
    col_keywords = st.text_input("Coluna Author Keywords", value="Author Keywords")

    # process dataframe
    with st.spinner("Processando dados (normalizando datas e termos)..."):
        # choose date column per row
        df = df_raw.copy()
        df['__date'] = df.apply(lambda r: choose_date(r, pub_col=col_pub, create_col=col_create), axis=1)
        # normalize period start
        df['__period_start'] = df['__date'].apply(lambda d: start_of_period(d, gran))
        # build terms_set from configured columns (if exist)
        term_cols = []
        if col_mesh in df.columns:
            term_cols.append(col_mesh)
        if col_keywords in df.columns:
            term_cols.append(col_keywords)
        if not term_cols:
            st.error("Nenhuma coluna de termos encontrada com os nomes informados. Ajuste os campos acima.")
            st.stop()

        def union_terms_from_row(row):
            s = set()
            for c in term_cols:
                for t in split_cell(row.get(c,'')):
                    if t:
                        s.add(t)
            return s

        df['__terms_set'] = df.apply(union_terms_from_row, axis=1)

        # filter out empty term sets
        df_nonempty = df[df['__terms_set'].map(lambda x: len(x) > 0)].copy()
        dropped = len(df) - len(df_nonempty)
        if dropped > 0:
            st.info(f"{dropped} registros sem termos foram ignorados para a análise.")

    # show sample
    st.subheader("Amostra de registros processados")
    sample = df_nonempty.head(10)[[col_pub, col_create, '__date','__period_start','__terms_set']]
    st.dataframe(sample)

    # build universal list of terms and document frequency per period
    st.info("Construindo séries temporais por termo (document frequency)...")
    # prepare periods range
    periods = sorted(df_nonempty['__period_start'].dropna().unique())
    if len(periods) == 0:
        st.error("Não há datas válidas para construir séries. Verifique as colunas de data.")
        st.stop()

    # Build mapping: term -> set of document ids per period
    # we'll use index as doc id
    term_period_docs = defaultdict(lambda: defaultdict(set))
    for idx, row in df_nonempty.iterrows():
        period = row['__period_start']
        doc_id = idx
        for term in row['__terms_set']:
            if term in custom_stop_terms:
                continue
            term_period_docs[term][period].add(doc_id)

    # Build DataFrame: rows periods (sorted) x columns terms with document frequency
    all_terms = sorted(term_period_docs.keys())
    if len(all_terms) == 0:
        st.error("Nenhum termo relevante encontrado após remoção de termos genéricos.")
        st.stop()

    # Create period index - ensure continuous range between min and max
    min_p = min(periods)
    max_p = max(periods)
    if gran == 'Monthly':
        period_index = pd.date_range(start=min_p, end=max_p, freq='MS')  # month start
    elif gran == 'Quarterly':
        period_index = pd.date_range(start=min_p, end=max_p, freq='QS')  # quarter start
    else:  # Yearly
        period_index = pd.date_range(start=min_p, end=max_p, freq='YS')

    # Build term x period DF
    data = {}
    for term in all_terms:
        counts = []
        for p in period_index:
            docs = term_period_docs[term].get(pd.Timestamp(p), set())
            counts.append(len(docs))
        data[term] = counts
    ts_df = pd.DataFrame(data, index=period_index)

    st.success(f"Séries construídas: {len(all_terms)} termos × {len(period_index)} períodos ({gran}).")
    st.dataframe(ts_df.iloc[-10: , : min(10, ts_df.shape[1])])  # preview last periods and first terms

    # ---------------- METRICAS ----------------
    st.subheader("Calculando métricas de tendência")
    # parameters per granularity
    if gran == 'Monthly':
        recency_window_periods = 12
        slope_recent_periods = 3
    elif gran == 'Quarterly':
        recency_window_periods = 4
        slope_recent_periods = 2
    else:  # Yearly
        recency_window_periods = 3
        slope_recent_periods = 1

    def compute_metrics_for_term(series):
        y = np.asarray(series, dtype=float)
        # slope via linear regression on log1p(y)
        # handle constant zero series
        if np.all(y == 0):
            slope = 0.0
        else:
            yy = np.log1p(y)
            x = np.arange(len(yy))
            # if too short, fallback to slope 0
            if len(yy) < 2:
                slope = 0.0
            else:
                # polyfit with degree 1
                try:
                    b = np.polyfit(x, yy, 1)[0]
                    slope = float(b)
                except Exception:
                    slope = 0.0
        # freq_recent for slope_score: sum of last slope_recent_periods
        if len(y) >= slope_recent_periods:
            freq_recent_for_slope = float(y[-slope_recent_periods:].sum())
        else:
            freq_recent_for_slope = float(y.sum())

        slope_score = slope * (np.log1p(freq_recent_for_slope) + 1e-9)

        # Z-score using last single period vs historical (exclude last)
        if len(y) >= 2:
            hist = y[:-1]
            mean_hist = hist.mean()
            std_hist = hist.std(ddof=0)
            last = y[-1]
            if std_hist == 0:
                z = 0.0
            else:
                z = float((last - mean_hist) / std_hist)
        else:
            z = 0.0

        # RecencyScore: sum of last recency_window_periods divided by sum of previous
        if len(y) <= recency_window_periods:
            recent = float(y.sum())
            histsum = 0.0
        else:
            recent = float(y[-recency_window_periods:].sum())
            histsum = float(y[:-recency_window_periods].sum())
        recency_score = recent / (histsum + 1.0)

        # also store basic stats
        total_docs = int(y.sum())
        last_period = int(y[-1]) if len(y)>0 else 0
        return {
            "slope": slope,
            "slope_score": slope_score,
            "z_score": z,
            "recency_score": recency_score,
            "total_docs": total_docs,
            "last_period": last_period
        }

    metrics = {}
    for term in ts_df.columns:
        metrics[term] = compute_metrics_for_term(ts_df[term].values)

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.index.name = 'term'
    metrics_df = metrics_df.reset_index()
    # ranking columns
    metrics_df['rank_slope'] = metrics_df['slope_score'].rank(ascending=False, method='min')
    metrics_df['rank_z'] = metrics_df['z_score'].rank(ascending=False, method='min')
    metrics_df['rank_recency'] = metrics_df['recency_score'].rank(ascending=False, method='min')

    # Top lists
    top_slope = metrics_df.sort_values('slope_score', ascending=False).head(top_n)
    top_z = metrics_df.sort_values('z_score', ascending=False).head(top_n)
    top_recency = metrics_df.sort_values('recency_score', ascending=False).head(top_n)

    # Consensus: terms that appear in at least 2 of the top lists
    top_terms_sets = [
        set(top_slope['term'].tolist()),
        set(top_z['term'].tolist()),
        set(top_recency['term'].tolist())
    ]
    # intersection counts
    from collections import Counter
    cnt = Counter()
    for s in top_terms_sets:
        for t in s:
            cnt[t]+=1
    consensus = [t for t,c in cnt.items() if c >= 2]
    strong_trends = sorted(consensus, key=lambda t: (metrics[t]['slope_score'] if t in metrics else 0), reverse=True)

    # Prepare final table
    final_cols = ['term','slope','slope_score','z_score','recency_score','total_docs','last_period']
    final_df = metrics_df[final_cols].sort_values('slope_score', ascending=False).reset_index(drop=True)

    st.success("Métricas calculadas.")

    # show top panels
    c1, c2, c3 = st.columns(3)
    c1.metric("Termos totais analisados", len(all_terms))
    c2.metric(f"Períodos (granularidade {gran})", len(period_index))
    c3.metric("Termos em consenso (≥2 métodos)", len(strong_trends))

    st.subheader("Top termos por métrica")
    tabs = st.tabs(["Slope (Regressão)","Z-score (Explosão)","RecencyScore (Híbrido)","Interseção (Tendências fortes)"])
    with tabs[0]:
        st.write(f"Top {top_n} por slope_score")
        st.dataframe(top_slope[['term','slope_score','slope','total_docs','last_period']].reset_index(drop=True))
    with tabs[1]:
        st.write(f"Top {top_n} por z_score")
        st.dataframe(top_z[['term','z_score','total_docs','last_period']].reset_index(drop=True))
    with tabs[2]:
        st.write(f"Top {top_n} por recency_score")
        st.dataframe(top_recency[['term','recency_score','total_docs','last_period']].reset_index(drop=True))
    with tabs[3]:
        st.write("Termos que aparecem em pelo menos 2 das listas Top N (consenso)")
        st.dataframe(pd.DataFrame({"term": strong_trends}).reset_index(drop=True))

    # allow user to inspect term series
    st.subheader("Explorar série temporal de um termo")
    term_selected = st.selectbox("Escolha termo", options=sorted(all_terms), index=0)
    if term_selected:
        series = ts_df[term_selected]
        st.write(f"Total docs: {int(series.sum())} — último período ({series.index[-1].strftime('%Y-%m-%d')}): {int(series.iloc[-1])}")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(series.index, series.values, marker='o')
        ax.set_title(f"Série temporal — {term_selected}")
        ax.set_xlabel("Período")
        ax.set_ylabel("Nº artigos (document frequency)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        # show small table
        st.dataframe(pd.DataFrame({"period":series.index, "count":series.values}).tail(36).reset_index(drop=True))

    # export final table
    st.subheader("Exportar resultados")
    buf = download_dataframe_as_csv(final_df)
    st.download_button("Baixar tabela completa (CSV)", data=buf, file_name="term_trends.csv", mime="text/csv")

    # optional: show wordcloud of top total_docs
    if st.checkbox("Mostrar WordCloud (top termos por total_docs)"):
        wc_terms = final_df.sort_values('total_docs', ascending=False).head(200).set_index('term')['total_docs'].to_dict()
        if len(wc_terms) > 0:
            wc = WordCloud(width=1200, height=600, background_color='white').generate_from_frequencies(wc_terms)
            fig2, ax2 = plt.subplots(figsize=(12,6))
            ax2.imshow(wc, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)
        else:
            st.info("Sem termos para wordcloud.")

else:
    st.info("Faça upload do CSV para começar a análise.")
