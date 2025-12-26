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

def create_sender_risk_heatmap(df: pd.DataFrame):
    """
    Create sender risk heatmap showing temporal patterns and anomaly clustering.
    X-axis: Time windows (10-second buckets)
    Y-axis: Top 15 most active senders
    Color: Anomaly score intensity (green → yellow → red)
    Shows: When and which senders exhibit high-risk behavior
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
            margin=dict(l=120, r=40, t=60, b=60)
        )
        return fig
    
    # Prepare data
    df = df.copy()
    
    # Calculate elapsed time in seconds from first transaction
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    min_time = df['ts'].min()
    df['elapsed_seconds'] = (df['ts'] - min_time).dt.total_seconds()
    
    # Create time windows (10-second buckets)
    time_bucket_size = 10  # seconds
    df['time_window'] = (df['elapsed_seconds'] // time_bucket_size).astype(int)
    
    # Get top 15 most active senders
    sender_counts = df['sender'].value_counts()
    top_senders = sender_counts.head(15).index.tolist()
    df_plot = df[df['sender'].isin(top_senders)].copy()
    
    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Building heatmap...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666666")
        )
        fig.update_layout(height=600, template="plotly_white")
        return fig
    
    # Aggregate by sender and time window - calculate mean anomaly score
    heatmap_data = df_plot.groupby(['sender', 'time_window']).agg({
        'score': 'mean',
        'amount': 'sum'
    }).reset_index()
    
    # Pivot to create heatmap matrix
    pivot_scores = heatmap_data.pivot(index='sender', columns='time_window', values='score')
    pivot_amounts = heatmap_data.pivot(index='sender', columns='time_window', values='amount')
    
    # Fill NaN with 0 (no transactions in that window)
    pivot_scores = pivot_scores.fillna(0)
    pivot_amounts = pivot_amounts.fillna(0)
    
    # Reindex to include all top senders (even if missing in some windows)
    pivot_scores = pivot_scores.reindex(top_senders, fill_value=0)
    pivot_amounts = pivot_amounts.reindex(top_senders, fill_value=0)
    
    # Create hover text with both score and amount
    hover_text = []
    for sender_idx, sender in enumerate(pivot_scores.index):
        hover_row = []
        for window_idx, window in enumerate(pivot_scores.columns):
            score = pivot_scores.iloc[sender_idx, window_idx]
            amount = pivot_amounts.iloc[sender_idx, window_idx]
            if score > 0:
                hover_row.append(
                    f"<b>{sender}</b><br>" +
                    f"Time: {window*10}-{(window+1)*10}s<br>" +
                    f"Avg Score: {score:.3f}<br>" +
                    f"Total Amount: ¥{amount:.2f}"
                )
            else:
                hover_row.append(f"<b>{sender}</b><br>Time: {window*10}-{(window+1)*10}s<br>No activity")
        hover_text.append(hover_row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_scores.values,
        x=[f"{int(w*10)}-{int((w+1)*10)}s" for w in pivot_scores.columns],
        y=pivot_scores.index.tolist(),
        colorscale=[
            [0.0, '#FFFFFF'],   # White for no activity
            [0.3, '#2ECC71'],   # Green for normal
            [0.5, '#F39C12'],   # Yellow for elevated
            [1.0, '#E74C3C']    # Red for high risk
        ],
        zmid=0.5,
        zmin=0,
        zmax=1,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(
            title=dict(
                text="Anomaly<br>Score",
                side="right"
            ),
            tickmode="linear",
            tick0=0,
            dtick=0.2,
            thickness=15,
            len=0.7
        )
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Sender Risk Heatmap: Temporal Pattern Detection',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        height=600,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        xaxis=dict(
            title='Time Window Since App Started',
            side='bottom',
            showgrid=False,
            tickangle=-45
        ),
        yaxis=dict(
            title='Sender ID',
            showgrid=False,
            autorange='reversed'  # Top sender at top
        ),
        margin=dict(l=100, r=80, t=80, b=100)
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
Real-Time Bank Transaction Anomaly Platform with Heatmap Pattern Detection

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
    
    # Heatmap Features
    with st.expander("ℹ️ Sender Risk Heatmap"):
        st.markdown("""
        **Heatmap Features:**
        - **X-axis**: Time windows (10-second intervals)
        - **Y-axis**: Top 15 most active senders
        - **Color**: Average anomaly score per window
        - **White**: No activity
        - **Green**: Normal behavior
        - **Yellow**: Elevated risk
        - **Red**: High risk
        
        **Key Patterns:**
        - Vertical red bands: Coordinated attacks
        - Horizontal red streaks: Persistent attackers
        - Diagonal patterns: Progressive attacks
        - Scattered reds: Isolated incidents
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
    - Sender risk heatmap visualization
    - Temporal coordination pattern detection
    - Multi-dimensional risk assessment
    
    **Technical Stack:**
    - Online Learning with Welford's Algorithm
    - Z-Score normalization per sender
    - Interactive Plotly heatmaps
    - 10-second time window aggregation
    
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
tab1, tab2 = st.tabs(["📊 Real-Time Monitor", "🔥 Sender Risk Heatmap"])

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
    ### Sender Risk Heatmap
    
    This heatmap reveals temporal patterns and coordination in transaction anomalies.
    Each cell shows a sender's average risk score during a 10-second time window.
    
    **Pattern Detection:**
    - **Red vertical bands**: Coordinated attack (multiple senders at same time)
    - **Red horizontal streaks**: Persistent attacker (one sender across time)
    - **Scattered red cells**: Isolated anomalies
    - **White cells**: No activity in that time window
    
    **Color Scale:**
    - White: No transactions
    - Green: Normal (score < 0.3)
    - Yellow: Elevated risk (0.3-0.6)
    - Red: High risk (> 0.6)
    """)
    
    # Sender risk heatmap
    heatmap_chart = create_sender_risk_heatmap(df)
    st.plotly_chart(heatmap_chart, use_container_width=True, key="heatmap_chart")
    
    # Summary statistics for top senders
    if not df.empty:
        st.markdown("### Top Sender Statistics")
        sender_stats = df.groupby('sender').agg({
            'amount': ['count', 'sum', 'mean'],
            'score': ['mean', 'max']
        }).round(2)
        sender_stats.columns = ['Tx Count', 'Total Amount', 'Avg Amount', 'Avg Score', 'Max Score']
        sender_stats = sender_stats.sort_values('Tx Count', ascending=False).head(15)
        st.dataframe(sender_stats, use_container_width=True)

# Footer
st.markdown("""
---
**TransGuardPlus v2.0** | Nippotica Corporation | Nippofin Business Unit | 
Powered by Isolation Forest ML + Sender Risk Heatmap Analysis
""")

#############################################
# AUTO-REFRESH LOGIC
#############################################

if st.session_state.is_running:
    generate_step(st.session_state.window_seconds)
    time.sleep(1.5)
    st.rerun()
