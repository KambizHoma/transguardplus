import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path

from transguardplus_core import TxSimulator, OnlineAnomalyModel, Featureizer, create_hourly_aggregation

#############################################
# PAGE CONFIGURATION
#############################################

st.set_page_config(
    page_title="TransGuardPlus - Real-Time Bank Transaction Anomaly Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#############################################
# SESSION STATE INITIALIZATION
#############################################

def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        # Load config
        cfg_path = Path("sample_config.json")
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
        else:
            cfg = {
                "seed": 42,
                "tx_per_second": 10,
                "num_senders": 50,
                "num_receivers": 50,
                "burst_prob": 0.10,
                "anomaly_prob": 0.15
            }
        
        st.session_state.cfg = cfg
        st.session_state.sim = TxSimulator(**cfg)
        st.session_state.feat = Featureizer()
        st.session_state.model = OnlineAnomalyModel()
        st.session_state.df = pd.DataFrame(columns=["ts", "sender", "receiver", "amount", "hour", "score"])
        st.session_state.is_running = False
        st.session_state.window_seconds = 300  # 5 minutes default for more data
        st.session_state.threshold = 0.65
        st.session_state.initialized = True

init_session_state()

#############################################
# CORE PROCESSING FUNCTIONS
#############################################

def score_batch(df_in: pd.DataFrame) -> pd.DataFrame:
    """Score a batch of transactions"""
    rows = []
    for row in df_in.itertuples(index=False):
        x, meta = st.session_state.feat.transform_row(row)
        s = st.session_state.model.score(x)
        rows.append({**meta, **x, "score": float(s)})
    
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        out = out.dropna(subset=["ts", "score"])
    return out


def append_and_window(df_new: pd.DataFrame, window_seconds: int):
    """Append new data and apply time window"""
    if df_new.empty:
        return
    
    all_df = pd.concat([st.session_state.df, df_new], ignore_index=True)
    horizon = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=window_seconds)
    st.session_state.df = all_df[all_df["ts"] >= horizon].reset_index(drop=True)


def generate_step(window_seconds: int):
    """Generate one batch of transactions"""
    batch = st.session_state.sim.generate_batch(n=st.session_state.cfg.get("tx_per_second", 10))
    scored = score_batch(batch)
    append_and_window(scored, window_seconds)


#############################################
# VISUALIZATION FUNCTIONS
#############################################

def create_animated_bubble_chart(df: pd.DataFrame):
    """
    Create animated bubble chart showing temporal clustering of anomalies.
    X-axis: Transaction amount (log scale)
    Y-axis: Sender anomaly score
    Size: Total volume per sender
    Color: Risk gradient (green/yellow/red)
    Animation: Hour progression
    """
    if df.empty or len(df) < 10:
        fig = go.Figure()
        fig.add_annotation(
            text="Accumulating data for temporal analysis...<br>Need at least 10 transactions",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666666")
        )
        fig.update_layout(
            height=600,
            template="plotly_white",
            margin=dict(l=60, r=40, t=60, b=60)
        )
        return fig
    
    # Create hourly aggregation
    agg_df = create_hourly_aggregation(df)
    
    if agg_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Processing hourly aggregations...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666666")
        )
        fig.update_layout(height=600, template="plotly_white")
        return fig
    
    # Determine color based on risk score
    def get_color(score):
        if score < 0.3:
            return '#2ECC71'  # Green - Normal
        elif score < 0.6:
            return '#F39C12'  # Yellow - Elevated
        else:
            return '#E74C3C'  # Red - High Risk
    
    agg_df['color'] = agg_df['mean_score'].apply(get_color)
    agg_df['risk_category'] = agg_df['mean_score'].apply(
        lambda x: 'Normal' if x < 0.3 else ('Elevated' if x < 0.6 else 'High Risk')
    )
    
    # Create figure with animation
    fig = px.scatter(
        agg_df,
        x='mean_amount',
        y='mean_score',
        size='total_amount',
        color='risk_category',
        animation_frame='hour',
        hover_name='sender',
        hover_data={
            'mean_amount': ':.2f',
            'mean_score': ':.3f',
            'total_amount': ':.2f',
            'tx_count': True,
            'risk_category': True,
            'hour': True
        },
        color_discrete_map={
            'Normal': '#2ECC71',
            'Elevated': '#F39C12',
            'High Risk': '#E74C3C'
        },
        labels={
            'mean_amount': 'Avg Transaction Amount (¥)',
            'mean_score': 'Anomaly Score',
            'total_amount': 'Total Volume (¥)',
            'tx_count': 'Transaction Count',
            'hour': 'Hour of Day'
        },
        size_max=60,
        range_y=[0, 1]
    )
    
    # Calculate reasonable X-axis range
    min_amt = agg_df['mean_amount'].min()
    max_amt = agg_df['mean_amount'].max()
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Temporal Pattern Analysis: Anomaly Clustering by Hour',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        height=600,
        template="plotly_white",
        showlegend=True,
        hovermode='closest',
        font=dict(family="Arial, sans-serif", size=12),
        xaxis=dict(
            type='log',
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e8e8e8',
            range=[np.log10(max(0.1, min_amt * 0.5)), np.log10(max_amt * 2)],
            tickformat='.0f',
            title='Avg Transaction Amount (¥)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e8e8e8',
            dtick=0.2,
            title='Anomaly Score'
        ),
        legend=dict(
            title="Risk Category",
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    # Add risk zone backgrounds
    fig.add_hrect(y0=0, y1=0.3, fillcolor="green", opacity=0.05, line_width=0)
    fig.add_hrect(y0=0.3, y1=0.6, fillcolor="yellow", opacity=0.05, line_width=0)
    fig.add_hrect(y0=0.6, y1=1.0, fillcolor="red", opacity=0.05, line_width=0)
    
    # Animation settings
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 800, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 400, 'easing': 'cubic-in-out'}
                    }]
                },
                {
                    'label': '❚❚ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 1.15,
            'xanchor': 'left',
            'yanchor': 'top'
        }]
    )
    
    return fig


def create_kde_plot(df: pd.DataFrame, threshold: float):
    """Create KDE (Kernel Density Estimation) distribution plot"""
    if df.empty or len(df) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text="Ready for data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#666666")
        )
        fig.update_layout(
            height=450,
            template="plotly_white",
            margin=dict(l=50, r=20, t=50, b=50)
        )
        return fig
    
    from scipy import stats
    
    scores = df['score'].values
    
    try:
        kde = stats.gaussian_kde(scores, bw_method=0.1)
    except:
        fig = go.Figure()
        fig.add_annotation(
            text="Buffering data for distribution",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#666666")
        )
        fig.update_layout(height=450, template="plotly_white")
        return fig
    
    x_range = np.linspace(0, 1, 300)
    density = kde(x_range)
    
    normal_mask = x_range < threshold
    alert_mask = x_range >= threshold
    
    fig = go.Figure()
    
    if normal_mask.any():
        fig.add_trace(go.Scatter(
            x=x_range[normal_mask],
            y=density[normal_mask],
            mode='lines',
            name='Normal',
            line=dict(color='#2ECC71', width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.3)',
            hovertemplate='Score: %{x:.3f}<br>Density: %{y:.2f}<extra></extra>'
        ))
    
    if alert_mask.any():
        fig.add_trace(go.Scatter(
            x=x_range[alert_mask],
            y=density[alert_mask],
            mode='lines',
            name='Alert',
            line=dict(color='#E74C3C', width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.3)',
            hovertemplate='Score: %{x:.3f}<br>Density: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        height=450,
        template="plotly_white",
        showlegend=True,
        xaxis_title="Anomaly Score",
        yaxis_title="Density",
        margin=dict(l=50, r=20, t=30, b=50),
        font=dict(family="Arial, sans-serif", size=11),
        xaxis=dict(
            range=[0, 1],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e8e8e8',
            fixedrange=True
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e8e8e8',
            fixedrange=True
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='lightgray',
            borderwidth=1
        )
    )
    
    return fig


def get_metrics(df: pd.DataFrame, threshold: float):
    """Calculate current metrics"""
    total_tx = len(df)
    alerts = int((df["score"] >= threshold).sum()) if not df.empty else 0
    mean_score = round(float(df["score"].mean()), 3) if not df.empty else 0.0
    
    if mean_score < 0.3:
        status = "✅ Normal"
    elif mean_score < 0.7:
        status = "⚠️ Elevated"
    else:
        status = "🚨 High Risk"
    
    return total_tx, alerts, mean_score, status


#############################################
# MAIN APP LAYOUT
#############################################

# Header
st.markdown("""
# TransGuardPlus 🛡️ 
Advanced Real-Time Bank Transaction Anomaly Platform with Temporal Pattern Analysis

**Nippotica Corporation** | Nippofin Business Unit | AI-Powered Surveillance  
""")

# Sidebar - Controls
with st.sidebar:
    st.markdown("### Controls")
    
    # Run/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Stream", use_container_width=True, type="primary"):
            st.session_state.is_running = True
            st.rerun()
    with col2:
        if st.button("⏸️ Stop Stream", use_container_width=True):
            st.session_state.is_running = False
            st.rerun()
    
    # Step and Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏭️ Step Once", use_container_width=True):
            generate_step(st.session_state.window_seconds)
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Data", use_container_width=True, type="secondary"):
            st.session_state.df = pd.DataFrame(columns=["ts", "sender", "receiver", "amount", "hour", "score"])
            st.session_state.feat = Featureizer()
            st.session_state.model = OnlineAnomalyModel()
            st.session_state.sim = TxSimulator(**st.session_state.cfg)
            st.rerun()
    
    # Stream status
    if st.session_state.is_running:
        st.success("🟢 Stream is RUNNING")
    else:
        st.info("⚪ Stream is STOPPED")
    
    st.markdown("---")
    st.markdown("### Settings")
    
    window_seconds = st.slider(
        "Time Window (seconds)",
        min_value=60,
        max_value=600,
        value=st.session_state.window_seconds,
        step=30,
        help="How much history to keep for temporal analysis"
    )
    st.session_state.window_seconds = window_seconds
    
    threshold = st.slider(
        "Alert Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.threshold,
        step=0.01,
        help="Score above this triggers alert"
    )
    st.session_state.threshold = threshold
    
    st.markdown("---")
    
    # Temporal Analysis Features
    with st.expander("ℹ️ Temporal Pattern Analysis"):
        st.markdown("""
        **Animated Bubble Chart Features:**
        - **X-axis**: Average transaction amount (log scale, ¥)
        - **Y-axis**: Anomaly score (0.0 to 1.0)
        - **Bubble Size**: Total transaction volume per sender
        - **Color**: Risk category (Green/Yellow/Red)
        - **Animation**: Hour-by-hour progression
        
        **Key Insights:**
        - Detect temporal clustering of anomalies
        - Track sender behavior evolution over time
        - Identify coordinated fraud patterns
        - Visualize amount-risk correlations
        """)
    
    # Distribution Features
    with st.expander("ℹ️ Distribution Analysis"):
        st.markdown("""
        **Density Curve:**
        - Shows probability distribution of anomaly scores
        - Green area: Normal transactions
        - Red area: Alert transactions above threshold
        - Helps identify score concentration patterns
        """)
    
    st.markdown("---")
    st.markdown("### Configuration")
    st.json(st.session_state.cfg)
    
    st.markdown("---")
    st.markdown("### About TransGuardPlus")
    
    st.markdown("""
    **Enhanced Features:**
    - Real-time anomaly detection (Isolation Forest)
    - Temporal pattern analysis with animated visualization
    - Hourly sender aggregation for clustering detection
    - Multi-dimensional risk assessment
    
    **Technical Stack:**
    - Online Learning with Welford's Algorithm
    - Z-Score normalization per sender
    - Interactive Plotly animations
    
    **Contact:** nippofin@nippotica.jp
    """)

# Main dashboard area
df = st.session_state.df.copy()
total_tx, alerts, mean_score, status = get_metrics(df, st.session_state.threshold)

# Metrics row
st.markdown("### Dashboard")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Transactions in Window", total_tx)
with col2:
    st.metric("🚨 Alerts", alerts)
with col3:
    st.metric("Mean Score", f"{mean_score:.3f}")
with col4:
    st.metric("Status", status)

# Create tabs for different views
tab1, tab2 = st.tabs(["📊 Real-Time Monitor", "🔄 Temporal Pattern Analysis"])

with tab1:
    st.markdown("### Anomaly Score Distribution")
    
    # KDE distribution plot
    kde_plot = create_kde_plot(df, st.session_state.threshold)
    st.plotly_chart(kde_plot, use_container_width=True, key="kde_plot")
    
    # Transaction table - Only Alerts
    st.markdown("### 🚨 Alert Transactions")
    alert_df = df[df["score"] >= st.session_state.threshold][["ts", "sender", "receiver", "amount", "score"]].copy()
    if not alert_df.empty:
        alert_df = alert_df.sort_values("score", ascending=False)
        alert_df["ts"] = alert_df["ts"].dt.strftime("%H:%M:%S")
        alert_df["score"] = alert_df["score"].round(3)
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(alert_df)} alert transaction(s) with score ≥ {st.session_state.threshold:.2f}")
    elif not df.empty:
        st.success("✅ No alerts in current window. All transactions are normal.")
    else:
        st.info("No transactions yet. Click 'Step Once' or 'Start Stream' to generate data.")

with tab2:
    st.markdown("""
    ### Temporal Pattern Analysis
    
    This animated bubble chart reveals how transaction anomalies cluster over time. 
    Each bubble represents a sender's aggregated behavior for that hour.
    
    **How to use:**
    - Press ▶ Play to watch hour-by-hour evolution
    - Larger bubbles indicate higher transaction volumes
    - Vertical position shows risk level
    - Observe if anomalies appear in coordinated bursts
    """)
    
    # Animated bubble chart
    bubble_chart = create_animated_bubble_chart(df)
    st.plotly_chart(bubble_chart, use_container_width=True, key="bubble_chart")
    
    # Summary statistics by hour
    if not df.empty:
        st.markdown("### Hourly Statistics")
        hourly_stats = df.groupby('hour').agg({
            'amount': ['count', 'sum', 'mean'],
            'score': ['mean', 'max']
        }).round(2)
        hourly_stats.columns = ['Tx Count', 'Total Amount', 'Avg Amount', 'Avg Score', 'Max Score']
        st.dataframe(hourly_stats, use_container_width=True)

# Footer
st.markdown("""
---
**TransGuardPlus v2.0** | Nippotica Corporation | Nippofin Business Unit | 
Powered by Isolation Forest ML + Temporal Pattern Analysis
""")

#############################################
# AUTO-REFRESH LOGIC
#############################################

if st.session_state.is_running:
    generate_step(st.session_state.window_seconds)
    time.sleep(1.5)
    st.rerun()
