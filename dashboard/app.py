"""
Real-time Sentiment Analysis Dashboard — Streamlit

Features:
  • Live single-text prediction with confidence gauge
  • Token-level attention heatmap
  • Batch CSV upload with downloadable results
  • Rolling prediction history log
  • Live API metrics panel

Run:
  streamlit run dashboard/app.py
"""

import json
import time
import random
from collections import deque
from datetime import datetime
from typing import Optional

import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ─── Config ──────────────────────────────────────────────────────────────────

API_BASE = os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000")
HISTORY_LIMIT = 100

# persistent HTTP session avoids per-request DNS + TCP handshake overhead
_session = requests.Session()

st.set_page_config(
    page_title="BERT Sentiment Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  body { background: #0d0d14; }
  .main { background: #0d0d14; }
  .metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .token-chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 8px;
    margin: 3px;
    font-family: monospace;
    font-size: 13px;
    font-weight: 600;
    color: #fff;
  }
  .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,40,0,0.3) !important;
    color: #eeeeff !important;
    border-radius: 10px !important;
  }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=HISTORY_LIMIT)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def call_predict(text: str, return_attention: bool = True) -> Optional[dict]:
    try:
        resp = _session.post(
            f"{API_BASE}/predict",
            json={"text": text, "return_attention": return_attention},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def call_batch(texts: list) -> Optional[dict]:
    try:
        resp = _session.post(
            f"{API_BASE}/predict/batch",
            json={"texts": texts},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def call_health() -> Optional[dict]:
    try:
        return _session.get(f"{API_BASE}/health", timeout=3).json()
    except Exception:
        return None


def call_metrics() -> Optional[dict]:
    try:
        return _session.get(f"{API_BASE}/metrics", timeout=3).json()
    except Exception:
        return None


def confidence_gauge(confidence: float, label: str) -> go.Figure:
    color = "#22c55e" if label == "positive" else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 32, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#444"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 50], "color": "rgba(239,68,68,0.12)"},
                {"range": [50, 75], "color": "rgba(234,179,8,0.12)"},
                {"range": [75, 100], "color": "rgba(34,197,94,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": confidence * 100},
        },
        title={"text": f"{label.upper()}", "font": {"size": 14, "color": "#aaa"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=40, b=0, l=20, r=20),
        font_color="#eeeeff",
    )
    return fig


def attention_heatmap(tokens: list, attention: list) -> go.Figure:
    if not tokens or not attention:
        return go.Figure()
    seq = len(tokens)
    matrix = np.array(attention)[:seq, :seq] if np.array(attention).ndim == 2 else np.zeros((seq, seq))
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=tokens,
        y=tokens,
        colorscale=[[0, "#0d0d14"], [0.5, "#7f1d1d"], [1, "#FF2800"]],
        showscale=True,
        colorbar=dict(tickcolor="#444", outlinecolor="rgba(0,0,0,0)"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=340,
        xaxis=dict(tickfont=dict(size=11, family="monospace"), side="bottom"),
        yaxis=dict(tickfont=dict(size=11, family="monospace"), autorange="reversed"),
        font_color="#eeeeff",
    )
    return fig


def token_chips(tokens: list, attention_row: list) -> str:
    if not tokens:
        return ""
    row = attention_row[0] if attention_row and len(attention_row) > 0 else [0.5] * len(tokens)
    maxv = max(row) or 1.0
    chips = ""
    for tok, w in zip(tokens, row):
        alpha = 0.15 + 0.85 * (w / maxv)
        r = int(255 * alpha)
        chips += f'<span class="token-chip" style="background:rgba({r},40,0,{alpha:.2f})">{tok}</span>'
    return chips


def history_chart(history: deque) -> go.Figure:
    if len(history) < 2:
        return go.Figure()
    df = pd.DataFrame(list(history))
    colors = ["#22c55e" if l == "positive" else "#ef4444" for l in df["label"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["confidence"],
        mode="lines+markers",
        line=dict(color="#FF2800", width=2),
        marker=dict(color=colors, size=8, line=dict(color="#0d0d14", width=1)),
        name="confidence",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(t=10, b=30, l=40, r=10),
        xaxis=dict(showgrid=False, color="#444"),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#444"),
        font_color="#eeeeff",
        showlegend=False,
    )
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 BERT Sentiment")
    st.markdown("---")

    health = call_health()
    if health:
        status_color = "🟢" if health["status"] == "healthy" else "🟡"
        st.markdown(f"{status_color} **{health['status'].capitalize()}**")
        st.markdown(f"Device: `{health['device']}`")
        st.markdown(f"Uptime: `{health['uptime_seconds']}s`")
        st.markdown(f"Predictions: `{health['total_predictions']:,}`")
    else:
        st.markdown("🔴 **API Offline**")
        st.caption(f"Trying: `{API_BASE}`")

    st.markdown("---")
    api_url = st.text_input("API URL", value=API_BASE)
    if api_url != API_BASE:
        API_BASE = api_url

    st.markdown("---")
    st.markdown("**Model:** `bert-base-uncased`")
    st.markdown("**Task:** SST-2 Binary Sentiment")
    st.markdown("**Accuracy:** `94.3%`")
    st.markdown("**F1:** `0.943`")

# ─── Main ─────────────────────────────────────────────────────────────────────

st.markdown("## 🧠 BERT Sentiment Analysis")
st.markdown("Real-time NLP inference powered by fine-tuned `bert-base-uncased`")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Live Analysis", "📂 Batch Upload", "📊 Metrics"])

# ─── Tab 1: Live Analysis ─────────────────────────────────────────────────────

with tab1:
    # seed text_area from session_state so example buttons can pre-fill it
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    col_input, col_result = st.columns([1.2, 1])

    with col_input:
        st.markdown("#### Input Text")
        text_input = st.text_area(
            "",
            height=140,
            placeholder="Type or paste any text here...",
            label_visibility="collapsed",
            key="input_text",
        )
        show_attention = st.checkbox("Show attention heatmap", value=True)
        run_btn = st.button("⚡ Analyze Sentiment", use_container_width=True, type="primary")

        st.markdown("**Quick examples:**")
        examples = [
            "The movie was absolutely brilliant, a masterpiece of storytelling!",
            "Terrible experience. Wasted my time completely.",
            "It was okay, nothing special but not bad either.",
            "Breakthrough performance by the entire cast, emotionally devastating.",
        ]
        for ex in examples:
            label = f"_{ex[:55]}..._" if len(ex) > 55 else f"_{ex}_"
            if st.button(label, use_container_width=True, key=f"ex_{ex[:20]}"):
                st.session_state["input_text"] = ex
                st.rerun()

    with col_result:
        st.markdown("#### Result")
        if run_btn and text_input.strip():
            with st.spinner("Running inference..."):
                resp = call_predict(text_input.strip(), return_attention=show_attention)
            if resp:
                r = resp["result"]
                st.plotly_chart(
                    confidence_gauge(r["confidence"], r["label"]),
                    use_container_width=True,
                )
                st.metric("Latency", f"{resp['latency_ms']} ms")

                p = r["probabilities"]
                prob_df = pd.DataFrame({"label": list(p.keys()), "probability": list(p.values())})
                st.bar_chart(prob_df.set_index("label"), height=120)

                st.session_state.history.appendleft({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "text": text_input[:60],
                    "label": r["label"],
                    "confidence": r["confidence"],
                })
        else:
            st.info("Enter text and click Analyze to see results.")

    if show_attention and run_btn and text_input.strip():
        resp2 = call_predict(text_input.strip(), return_attention=True)
        if resp2 and resp2["result"].get("tokens"):
            r2 = resp2["result"]
            st.markdown("#### Token Attention Weights")
            tokens = r2["tokens"]
            attn = r2.get("attention", [])

            chips_html = token_chips(tokens, attn)
            st.markdown(chips_html, unsafe_allow_html=True)

            if attn and len(attn) > 1:
                st.plotly_chart(attention_heatmap(tokens, attn), use_container_width=True)

    if st.session_state.history:
        st.markdown("#### Prediction History")
        st.plotly_chart(history_chart(st.session_state.history), use_container_width=True)
        hist_df = pd.DataFrame(list(st.session_state.history))
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ─── Tab 2: Batch Upload ──────────────────────────────────────────────────────

with tab2:
    st.markdown("#### Upload CSV for Batch Inference")
    st.caption("CSV must contain a `text` column. Max 500 rows per upload.")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must have a `text` column.")
        else:
            st.dataframe(df.head(5), use_container_width=True)
            if st.button("🚀 Run Batch Inference", type="primary"):
                texts = df["text"].dropna().astype(str).tolist()[:500]
                progress = st.progress(0)
                all_results = []
                chunk_size = 32
                for i in range(0, len(texts), chunk_size):
                    chunk = texts[i : i + chunk_size]
                    resp = call_batch(chunk)  # uses _session internally
                    if resp:
                        all_results.extend(resp["results"])
                    progress.progress(min(1.0, (i + chunk_size) / len(texts)))

                results_df = pd.DataFrame([
                    {"text": r["text"], "label": r["label"], "confidence": r["confidence"]}
                    for r in all_results
                ])
                st.success(f"Processed {len(results_df)} texts.")

                fig = px.pie(
                    results_df, names="label",
                    color_discrete_map={"positive": "#22c55e", "negative": "#ef4444"},
                    title="Sentiment Distribution",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#eeeeff")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(results_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download Results",
                    results_df.to_csv(index=False),
                    "sentiment_results.csv",
                    "text/csv",
                )

# ─── Tab 3: Metrics ───────────────────────────────────────────────────────────

with tab3:
    st.markdown("#### Live API Metrics")
    if st.button("🔄 Refresh Metrics"):
        st.rerun()

    m = call_metrics()
    if m:
        lat = m.get("latency", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Latency", f"{lat.get('mean_ms', '-')} ms")
        c2.metric("P95 Latency", f"{lat.get('p95_ms', '-')} ms")
        c3.metric("P99 Latency", f"{lat.get('p99_ms', '-')} ms")
        c4.metric("Total Predictions", f"{m.get('total_predictions', 0):,}")
    else:
        st.warning("Could not reach API metrics endpoint.")
