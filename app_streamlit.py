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

def create_animated_sender_timeline(df: pd.DataFrame):
    """
    Create animated sender risk timeline showing temporal clustering and coordination.
    X-axis: Elapsed time since app started (seconds)
    Y-axis: Sender IDs (top 15 most active)
    Dots: Individual transactions colored by anomaly score, sized by amount
    Animation: Cumulative reveal with playhead moving left-to-right
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
    
    # Get top 15 most active senders
    sender_counts = df['sender'].value_counts()
    top_senders = sender_counts.head(15).index.tolist()
    df_plot = df[df['sender'].isin(top_senders)].copy()
    
    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Building sender timeline...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666666")
        )
        fig.update_layout(height=600, template="plotly_white")
        return fig
    
    # Create time buckets for animation (10-second intervals)
    max_elapsed = df_plot['elapsed_seconds'].max()
    time_bucket_size = 10  # seconds
    df_plot['time_bucket'] = (df_plot['elapsed_seconds'] // time_bucket_size) * time_bucket_size
    
    # Assign numeric Y positions to senders (for better control)
    sender_to_y = {sender: idx for idx, sender in enumerate(reversed(top_senders))}
    df_plot['sender_y'] = df_plot['sender'].map(sender_to_y)
    
    # Determine color based on anomaly score
    def get_color(score):
        if score < 0.3:
            return '#2ECC71'  # Green - Normal
        elif score < 0.6:
            return '#F39C12'  # Yellow - Elevated
        else:
            return '#E74C3C'  # Red - High Risk
    
    df_plot['color'] = df_plot['score'].apply(get_color)
    df_plot['risk_category'] = df_plot['score'].apply(
        lambda x: 'Normal' if x < 0.3 else ('Elevated' if x < 0.6 else 'High Risk')
    )
    
    # Scale transaction amounts for bubble size (log scale for better visibility)
    df_plot['size'] = np.log1p(df_plot['amount']) * 3 + 5  # Min size 5, scaled by log
    
    # For cumulative animation, we need to prepare data differently
    # Create a cumulative dataset for each time bucket
    time_buckets = sorted(df_plot['time_bucket'].unique())
    
    # Prepare cumulative data for each time bucket
    cumulative_data = []
    for bucket in time_buckets:
        bucket_df = df_plot[df_plot['time_bucket'] <= bucket].copy()
        bucket_df['animation_frame'] = bucket
        cumulative_data.append(bucket_df)
    
    df_animated = pd.concat(cumulative_data, ignore_index=True)
    
    # Create animated scatter plot with cumulative data
    fig = px.scatter(
        df_animated,
        x='elapsed_seconds',
        y='sender_y',
        size='size',
        color='risk_category',
        animation_frame='animation_frame',
        hover_name='sender',
        hover_data={
            'elapsed_seconds': ':.1f',
            'amount': ':.2f',
            'score': ':.3f',
            'receiver': True,
            'risk_category': True,
            'sender_y': False,
            'animation_frame': False,
            'time_bucket': False,
            'size': False,
            'color': False
        },
        color_discrete_map={
            'Normal': '#2ECC71',
            'Elevated': '#F39C12',
            'High Risk': '#E74C3C'
        },
        labels={
            'elapsed_seconds': 'Elapsed Time (seconds)',
            'sender_y': 'Sender',
            'amount': 'Amount (¥)',
            'score': 'Anomaly Score',
            'receiver': 'Receiver'
        },
        size_max=30,
        range_x=[0, max(max_elapsed + 10, 60)],
        range_y=[-0.5, len(top_senders) - 0.5]
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Sender Risk Timeline: Temporal Coordination Detection',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        height=600,
        template="plotly_white",
        showlegend=True,
        hovermode='closest',
        font=dict(family="Arial, sans-serif", size=11),
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e8e8e8',
            title='Elapsed Time Since App Started (seconds)'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e8e8e8',
            tickmode='array',
            tickvals=list(range(len(top_senders))),
            ticktext=list(reversed(top_senders)),
            title='Sender ID'
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
        margin=dict(l=120, r=40, t=80, b=60)
    )
    
    # Animation settings - 0.8 seconds per frame, cumulative
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
                        'transition': {'duration': 300, 'easing': 'linear'}
                    }]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.05,
            'y': 1.12,
            'xanchor': 'left',
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'y': -0.1,
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Time: ',
                'visible': True,
                'xanchor': 'right',
                'suffix': 's'
            },
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.05
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
Real-Time Bank Transaction Anomaly Platform with Sender Coordination Detection

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
    with st.expander("ℹ️ Sender Risk Timeline"):
        st.markdown("""
        **Animated Timeline Features:**
        - **X-axis**: Elapsed time since app started (seconds)
        - **Y-axis**: Top 15 most active senders
        - **Dot Color**: Risk category (Green/Yellow/Red)
        - **Dot Size**: Transaction amount
        - **Animation**: Cumulative reveal (0.8s per frame)
        
        **Key Patterns to Watch:**
        - Horizontal red lines: Persistent attacker
        - Vertical red clusters: Coordinated attack
        - Burst patterns: Sudden fraud attempts
        - Timeline evolution: Risk progression over session
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
    - Animated sender risk timeline
    - Temporal coordination pattern detection
    - Multi-dimensional risk assessment
    
    **Technical Stack:**
    - Online Learning with Welford's Algorithm
    - Z-Score normalization per sender
    - Interactive Plotly animations
    - Cumulative timeline reveal (0.8s frames)
    
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
tab1, tab2 = st.tabs(["📊 Real-Time Monitor", "⏱️ Sender Risk Timeline"])

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
    ### Sender Risk Timeline
    
    This animated timeline reveals temporal coordination patterns in transaction anomalies.
    Each horizontal lane represents a sender. Watch for patterns:
    
    - **Horizontal red clusters**: One sender attacking repeatedly
    - **Vertical red clusters**: Multiple senders coordinating attacks at same time
    - **Burst patterns**: Sudden appearance of many transactions
    
    **How to use:**
    - Press **▶ Play** to watch the timeline unfold from app start
    - **⏸ Pause** to freeze at any moment
    - **Drag the slider** to jump to specific time points
    - **Hover over dots** to see transaction details
    """)
    
    # Animated sender timeline
    timeline_chart = create_animated_sender_timeline(df)
    st.plotly_chart(timeline_chart, use_container_width=True, key="timeline_chart")
    
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
Powered by Isolation Forest ML + Sender Risk Timeline Analysis
""")

#############################################
# AUTO-REFRESH LOGIC
#############################################

if st.session_state.is_running:
    generate_step(st.session_state.window_seconds)
    time.sleep(1.5)
    st.rerun()
