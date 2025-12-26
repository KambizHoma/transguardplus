# TransGuardPlus 🛡️

Advanced Real-Time Bank Transaction Anomaly Platform with Temporal Pattern Analysis

**Developed by:** Nippotica Corporation | Nippofin Business Unit  
**Version:** 2.0  
**Target Audience:** Japanese Financial Institutions (Banks, Securities Firms)

---

## Overview

TransGuardPlus is an enhanced version of TransGuard featuring **animated temporal pattern analysis** to detect coordinated fraud and anomaly clustering in bank transaction streams.

### Key Enhancements Over TransGuard v1.0

- ✨ **Animated Bubble Chart**: Hour-by-hour visualization of anomaly clustering
- 📊 **Temporal Pattern Analysis**: Detect coordinated fraud patterns over time
- 🎯 **Multi-dimensional Risk View**: Amount, score, volume, and time correlation
- 🔄 **Interactive Exploration**: Play/pause animation controls with Plotly

---

## Features

### 1. Real-Time Monitoring (Tab 1)
- Live transaction stream simulation
- Kernel Density Estimation (KDE) distribution plot
- Real-time alert table
- Configurable alert threshold

### 2. Temporal Pattern Analysis (Tab 2)
- **Animated Bubble Chart** showing:
  - **X-axis**: Average transaction amount (log scale, ¥)
  - **Y-axis**: Anomaly score (0.0 to 1.0)
  - **Bubble Size**: Total transaction volume per sender
  - **Color**: Risk category (Green/Yellow/Red)
  - **Animation**: Hour-by-hour progression

- **Hourly Statistics Table**:
  - Transaction count per hour
  - Total/average amounts
  - Average/maximum scores

### 3. Core Technology
- **Isolation Forest**: Online learning anomaly detection
- **Welford's Algorithm**: Efficient streaming statistics
- **Z-Score Normalization**: Per-sender feature engineering
- **Hourly Aggregation**: Temporal clustering detection

---

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/transguardplus.git
cd transguardplus

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app_streamlit.py
```

### Streamlit Cloud Deployment

See **Deployment Instructions** section below.

---

## Usage

### Starting the Application

1. Click **▶️ Start Stream** to begin transaction simulation
2. Monitor real-time anomalies in the **Real-Time Monitor** tab
3. Switch to **Temporal Pattern Analysis** tab to view animated bubble chart
4. Press **▶ Play** button on the animation to watch hour-by-hour evolution

### Configuration

Adjust settings in the sidebar:
- **Time Window**: History retention (60-600 seconds)
- **Alert Threshold**: Anomaly score threshold (0.0-1.0)

Modify `sample_config.json` for simulation parameters:
- `tx_per_second`: Transaction generation rate
- `num_senders`/`num_receivers`: Entity pool size
- `burst_prob`: Probability of transaction bursts
- `anomaly_prob`: Probability of anomalous transactions

---

## Technical Architecture

### File Structure

```
transguardplus/
├── app_streamlit.py              # Main Streamlit application
├── transguardplus_core.py        # Core logic and models
├── requirements.txt              # Python dependencies
├── sample_config.json            # Simulation configuration
└── README.md                     # This file
```

### Core Components

**transguardplus_core.py:**
- `TxSimulator`: Generates realistic transaction streams with anomalies
- `Featureizer`: Extracts features (Z-scores, sender statistics)
- `OnlineAnomalyModel`: Isolation Forest with online learning
- `create_hourly_aggregation()`: Aggregates data for temporal analysis

**app_streamlit.py:**
- Session state management
- Real-time data processing pipeline
- Interactive Plotly visualizations
- Tab-based UI layout

---

## Deployment Instructions

### GitHub Desktop Workflow

#### Step 1: Create New Repository
1. Open **GitHub Desktop**
2. Click **File** → **New Repository**
3. Fill in:
   - **Name**: `transguardplus`
   - **Description**: `Advanced Real-Time Bank Transaction Anomaly Platform`
   - **Local Path**: Choose your working directory
   - **Initialize with README**: ✅ Checked
4. Click **Create Repository**

#### Step 2: Add Project Files
1. Copy all TransGuardPlus files to the repository folder:
   - `app_streamlit.py`
   - `transguardplus_core.py`
   - `requirements.txt`
   - `sample_config.json`
   - `README.md`

2. In GitHub Desktop, you'll see all files listed under "Changes"
3. Enter commit message: `Initial commit - TransGuardPlus v2.0`
4. Click **Commit to main**

#### Step 3: Publish to GitHub
1. Click **Publish repository** button
2. Verify repository name: `transguardplus`
3. Add description: `Advanced Real-Time Bank Transaction Anomaly Platform with Temporal Pattern Analysis`
4. Uncheck **Keep this code private** (or keep checked if preferred)
5. Click **Publish repository**

### Streamlit Cloud Deployment

#### Step 1: Access Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub account
3. Click **New app**

#### Step 2: Configure Deployment
1. **Repository**: Select `YOUR_USERNAME/transguardplus`
2. **Branch**: `main`
3. **Main file path**: `app_streamlit.py`
4. **App URL** (custom): `transguardplus` (or your preferred name)

#### Step 3: Deploy
1. Click **Deploy!**
2. Wait 2-3 minutes for deployment
3. Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

#### Step 4: Test Deployment
1. Visit the deployed URL
2. Click **▶️ Start Stream**
3. Verify both tabs work correctly
4. Test the animated bubble chart

---

## Future Enhancements

- [ ] Multi-currency support (USD, EUR, JPY)
- [ ] Export alert reports (PDF/Excel)
- [ ] Integration with real banking APIs
- [ ] Machine learning model versioning
- [ ] Historical replay mode
- [ ] Customizable risk zones

---

## Support

For questions or support:
- **Email**: nippofin@nippotica.jp
- **Documentation**: See sidebar tooltips in application

---

## License

Proprietary - Nippotica Corporation  
For demonstration and evaluation purposes only.

---

**Built with:** Python 3.10+ | Streamlit | Plotly | scikit-learn | pandas | NumPy | SciPy
