# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import sys
import os
from datetime import datetime

# ensure project src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.predict import predict_single, predict_batch, get_feature_importances, load_assets
from src.db import init_db, log_prediction, log_batch

st.set_page_config(page_title="Emotion Detector", layout="wide", page_icon="😊")

# Initialize DB
init_db()

# Top header
st.markdown(
    """
    <style>
    .big-title { font-size:32px; font-weight:700; }
    .sub { color: #6c757d; }
    .card { padding: 1rem; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); background: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<div class="big-title">Emotion Detection Studio 🎭</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Interactive TF-IDF NLP Emotion Classifier • Batch Predictions • Live Visualizations</div>', unsafe_allow_html=True)
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Emoji_u1f600.svg", width=70)

st.write("---")

# Load model info
_model, _vectorizer, _labels, _reverse_labels = load_assets()

# Layout: left panel for input, right panel for visualizations
left, right = st.columns([2,3])

# LEFT SIDE
with left:
    st.markdown("### Try a single prediction")
    text = st.text_area("Enter text to classify", height=140, placeholder="I am so excited today!")
    btn_predict = st.button("Predict 🎯")

    if btn_predict:
        if not text.strip():
            st.warning("Please enter text.")
        else:
            with st.spinner("Predicting..."):
                res = predict_single(text)
                log_prediction(res, source="single")

            st.success(f"**Predicted:** {res['pred_label']}")

            if res.get("probabilities"):
                probs = { _reverse_labels[i]: p for i,p in enumerate(res["probabilities"]) }
                prob_df = pd.DataFrame(list(probs.items()), columns=["emotion","probability"]) \
                            .sort_values("probability", ascending=False)

                fig = px.bar(
                    prob_df,
                    x="emotion",
                    y="probability",
                    title="Emotion Probabilities",
                    text_auto=".2f"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.balloons()

    st.write("---")
    st.markdown("### Batch Prediction (CSV)")
    uploaded = st.file_uploader("Upload CSV with a column named 'text'", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            if st.button("Run Batch Prediction"):
                with st.spinner("Predicting..."):
                    records = predict_batch(df["text"].astype(str).tolist())
                    log_batch(records, source="batch")

                    out_df = pd.DataFrame(records)

                    # Expand probabilities
                    prob_cols = [f"prob_{_reverse_labels[i]}" for i in range(len(_labels))]
                    if out_df["probabilities"].notnull().any():
                        out_df[prob_cols] = pd.DataFrame(out_df["probabilities"].tolist(), index=out_df.index)

                    out_df.drop(columns=["probabilities"], inplace=True)

                st.success("Batch prediction complete!")
                st.dataframe(out_df.head(50))

                # Download link
                csv_bytes = out_df.to_csv(index=False).encode()
                b64 = base64.b64encode(csv_bytes).decode()
                st.markdown(
                    f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Results</a>',
                    unsafe_allow_html=True
                )

# RIGHT SIDE
with right:
    st.markdown("## Visualizations")

    viz_tab, importance_tab, db_tab = st.tabs(["Emotion Stats", "TF-IDF Importances", "DB Insights"])

    # Emotion stats
    with viz_tab:
        st.markdown("### Logged Emotion Distribution")

        import sqlite3
        conn = sqlite3.connect("predictions.db")

        try:
            stats_df = pd.read_sql_query(
                "SELECT predicted_label, COUNT(*) as cnt FROM predictions GROUP BY predicted_label",
                conn
            )
        finally:
            conn.close()

        if stats_df.empty:
            st.info("No predictions yet.")
        else:
            pie_fig = px.pie(
                stats_df,
                names="predicted_label",
                values="cnt",
                title="Prediction Share",
                hole=0.4
            )
            st.plotly_chart(pie_fig, use_container_width=True)

            bar_fig = px.bar(
                stats_df.sort_values("cnt", ascending=False),
                x="predicted_label",
                y="cnt",
                title="Count of Predictions per Emotion",
                text_auto=True
            )
            st.plotly_chart(bar_fig, use_container_width=True)

    # TF-IDF importances
    with importance_tab:
        st.markdown("### Top Words per Emotion")

        top_n = st.slider("Number of top words", 5, 40, 12)
        importances = get_feature_importances(top_n=top_n)
        selected = st.selectbox("Select emotion", list(importances.keys()))

        df_imp = pd.DataFrame(importances[selected], columns=["word", "weight"])

        fig_imp = px.bar(
            df_imp.sort_values("weight"),
            x="weight",
            y="word",
            orientation="h",
            title=f"Top {top_n} Words for '{selected}'",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # DB logs
    with db_tab:
        st.markdown("### Recent logged predictions")
        conn = sqlite3.connect("predictions.db")
        try:
            df_db = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 200",
                conn
            )
        finally:
            conn.close()

        st.dataframe(df_db)
