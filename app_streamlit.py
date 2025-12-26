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

def create_sender_risk_bars(df: pd.DataFrame):
    """
    Create horizontal bar chart showing top senders by average anomaly score.
    Simple, clear visualization identifying problematic senders.
    """
    if df.empty or len(df) < 10:
        fig = go.Figure()
        fig.add_annotation(
            text="Accumulating data for sender analysis...<br>Need at least 10 transactions",
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
    
    # Aggregate by sender
    sender_stats = df.groupby('sender').agg({
        'score': ['mean', 'max', 'count'],
        'amount': ['sum', 'mean']
    }).reset_index()
    
    sender_stats.columns = ['sender', 'avg_score', 'max_score', 'tx_count', 'total_amount', 'avg_amount']
    
    # Get top 15 by average score
    top_senders = sender_stats.nlargest(15, 'avg_score')
    
    # Determine color based on average score
    def get_color(score):
        if score < 0.3:
            return '#2ECC71'  # Green - Normal
        elif score < 0.6:
            return '#F39C12'  # Yellow - Elevated
        else:
            return '#E74C3C'  # Red - High Risk
    
    top_senders['color'] = top_senders['avg_score'].apply(get_color)
    top_senders['risk_category'] = top_senders['avg_score'].apply(
        lambda x: 'Normal' if x < 0.3 else ('Elevated' if x < 0.6 else 'High Risk')
    )
    
    # Sort by score for better visualization
    top_senders = top_senders.sort_values('avg_score', ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    for category, color in [('High Risk', '#E74C3C'), ('Elevated', '#F39C12'), ('Normal', '#2ECC71')]:
        cat_data = top_senders[top_senders['risk_category'] == category]
        if not cat_data.empty:
            fig.add_trace(go.Bar(
                y=cat_data['sender'],
                x=cat_data['avg_score'],
                orientation='h',
                name=category,
                marker=dict(color=color),
                text=cat_data['avg_score'].round(3),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>' +
                              'Avg Score: %{x:.3f}<br>' +
                              'Max Score: %{customdata[0]:.3f}<br>' +
                              'Transactions: %{customdata[1]}<br>' +
                              'Total Amount: ¥%{customdata[2]:.2f}<br>' +
                              '<extra></extra>',
                customdata=cat_data[['max_score', 'tx_count', 'total_amount']].values
            ))
    
    # Add risk zone backgrounds
    fig.add_vrect(x0=0, x1=0.3, fillcolor="green", opacity=0.05, line_width=0)
    fig.add_vrect(x0=0.3, x1=0.6, fillcolor="yellow", opacity=0.05, line_width=0)
    fig.add_vrect(x0=0.6, x1=1.0, fillcolor="red", opacity=0.05, line_width=0)
    
    # Add threshold lines
    fig.add_vline(x=0.3, line=dict(color='#2ECC71', width=1, dash='dash'), 
                  annotation_text="Normal", annotation_position="top")
    fig.add_vline(x=0.6, line=dict(color='#F39C12', width=1, dash='dash'),
                  annotation_text="Elevated", annotation_position="top")
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Top 15 Senders by Average Anomaly Score',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        height=600,
        template="plotly_white",
        showlegend=True,
        barmode='overlay',
        font=dict(family="Arial, sans-serif", size=11),
        xaxis=dict(
            title='Average Anomaly Score',
            range=[0, 1],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='#e8e8e8',
            dtick=0.1
        ),
        yaxis=dict(
            title='Sender ID',
            showgrid=False
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
        margin=dict(l=100, r=100, t=80, b=60)
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
Real-Time Bank Transaction Anomaly Platform with Sender Risk Analysis

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
    
    # Sender Risk Analysis Features
    with st.expander("ℹ️ Sender Risk Analysis"):
        st.markdown("""
        **Bar Chart Features:**
        - **Top 15 senders** ranked by average anomaly score
        - **Horizontal bars** - longer = higher risk
        - **Color coding**: Green/Yellow/Red by risk category
        - **Risk zones** - visual background thresholds
        
        **Detailed Statistics Table:**
        - Transaction count and amounts
        - Score statistics (avg, max, min, std dev)
        - Risk category classification
        - Sortable by any column
        
        **Use Cases:**
        - Identify bad actors quickly
        - Prioritize investigation efforts
        - Track sender behavior patterns
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
    - Breathing dots distribution visualization
    - Sender risk ranking and analysis
    - Comprehensive sender statistics
    
    **Technical Stack:**
    - Online Learning with Welford's Algorithm
    - Z-Score normalization per sender
    - Interactive Plotly visualizations
    - Statistical aggregation and ranking
    
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
tab1, tab2 = st.tabs(["📊 Real-Time Monitor", "👥 Sender Risk Analysis"])

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

    
    # Sender risk bar chart
    bars_chart = create_sender_risk_bars(df)
    st.plotly_chart(bars_chart, use_container_width=True, key="bars_chart")
 
with tab2:
    st.markdown("""
    ### Sender Risk Analysis
    
    Identify the most problematic senders based on their average anomaly scores.
    This visualization ranks senders by risk level to quickly identify bad actors.
    
    **Risk Categories:**
    - 🟢 **Green (< 0.3)**: Normal behavior
    - 🟡 **Yellow (0.3-0.6)**: Elevated risk - monitor closely
    - 🔴 **Red (> 0.6)**: High risk - investigate immediately
    
    **How to Use:**
    - Longer bars = higher average risk
    - Hover over bars for detailed statistics
    - Check the table below for full transaction breakdown
    """)
    
    # Detailed sender statistics table
    if not df.empty:
        st.markdown("### 📋 Detailed Sender Statistics")
        
        sender_details = df.groupby('sender').agg({
            'amount': ['count', 'sum', 'mean', 'min', 'max'],
            'score': ['mean', 'max', 'min', 'std']
        }).round(3)
        
        sender_details.columns = [
            'Tx Count', 'Total Amount', 'Avg Amount', 'Min Amount', 'Max Amount',
            'Avg Score', 'Max Score', 'Min Score', 'Score StdDev'
        ]
        
        # Sort by average score descending
        sender_details = sender_details.sort_values('Avg Score', ascending=False)
        
        # Add risk category
        sender_details['Risk Category'] = sender_details['Avg Score'].apply(
            lambda x: '🔴 High Risk' if x >= 0.6 else ('🟡 Elevated' if x >= 0.3 else '🟢 Normal')
        )
        
        # Reorder columns to put risk category first
        cols = ['Risk Category'] + [col for col in sender_details.columns if col != 'Risk Category']
        sender_details = sender_details[cols]
        
        st.dataframe(sender_details, use_container_width=True)
        st.caption(f"Showing statistics for all {len(sender_details)} senders in current time window")


# Footer
st.markdown("""
---
**TransGuardPlus v2.0** | Nippotica Corporation | Nippofin Business Unit | 
Powered by Isolation Forest ML + Sender Risk Analysis
""")

#############################################
# AUTO-REFRESH LOGIC
#############################################

if st.session_state.is_running:
    generate_step(st.session_state.window_seconds)
    time.sleep(1.5)
    st.rerun()
