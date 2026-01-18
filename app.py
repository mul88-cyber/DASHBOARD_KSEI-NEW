# ==============================================================================
# 📦 1) IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import library Google
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ 2) KONFIGURASI DASHBOARD & G-DRIVE
# ==============================================================================
st.set_page_config(
    page_title="KSEI Bandarmology PRO",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# --- 🔥 FIX CSS (KSEI STYLE PROPER - LIGHT THEME) ---
st.markdown("""
<style>
    /* 1. FORCE LIGHT THEME VARIABLES */
    :root {
        --primary-color: #4318FF;
        --background-color: #F4F7FE;
        --secondary-background-color: #FFFFFF;
        --text-color: #2B3674;
        --font: 'DM Sans', sans-serif;
    }

    /* 2. Main Background & Text Reset */
    .stApp {
        background-color: #F4F7FE;
        color: #2B3674;
    }
    
    /* 3. Sidebar Styling (Full White & Clean) */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 14px 14px 40px rgba(112, 144, 176, 0.08);
        border-right: none;
    }
    
    /* Fix: Memastikan semua teks di sidebar terlihat (Dark Blue) */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #2B3674 !important;
        font-weight: 600;
    }
    
    /* 4. Widget Inputs Styling */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E5F2 !important;
        color: #2B3674 !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] span, input.st-ac {
        color: #2B3674 !important;
    }
    
    /* 5. Custom Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4318FF 0%, #868CFF 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
        box-shadow: 0px 4px 10px rgba(67, 24, 255, 0.2);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 15px rgba(67, 24, 255, 0.3);
        border: none;
        color: white;
    }

    /* 6. Card Styling */
    .css-card {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.12);
        margin-bottom: 24px;
        border: none;
    }
    
    /* 7. Header Banner */
    .header-banner {
        background: linear-gradient(86.88deg, #4318FF 0%, #868CFF 100%);
        border-radius: 20px;
        padding: 30px 40px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0px 18px 40px rgba(112, 144, 176, 0.2);
    }
    .header-title { font-size: 32px; font-weight: 700; margin-bottom: 8px; }
    .header-subtitle { font-size: 16px; font-weight: 500; opacity: 0.9; }
    
    /* 8. Fix Label Visibility Globally */
    label {
        color: #2B3674 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    /* 9. Metric Cards Clean up */
    div[data-testid="stMetricValue"] {
        color: #2B3674 !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #A3AED0 !important;
    }
    
    /* 10. Table/Dataframe Styling */
    div[data-testid="stDataFrame"] {
        background-color: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.05);
    }
    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #2B3674;
        margin-bottom: 20px;
    }
    
    /* 11. Badge Styling */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin: 2px;
    }
    .badge-high { background-color: #D6F5E3; color: #0D9D58; }
    .badge-medium { background-color: #FFF4E5; color: #FF9800; }
    .badge-low { background-color: #FFE5E5; color: #FF3B30; }
    .badge-stealth { background-color: #E5F3FF; color: #0066CC; }
    .badge-coordinated { background-color: #F0E5FF; color: #7B1FA2; }
</style>
""", unsafe_allow_html=True)

# --- KONFIGURASI G-DRIVE ---
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP"
FILE_NAME = "KSEI_Shareholder_Processed.csv"

# --- KONFIGURASI KATEGORI ---
OWNERSHIP_COLS = [
    'Local IS', 'Local CP', 'Local PF', 'Local IB', 'Local ID', 'Local MF', 'Local SC', 'Local FD', 'Local OT',
    'Foreign IS', 'Foreign CP', 'Foreign PF', 'Foreign IB', 'Foreign ID', 'Foreign MF', 'Foreign SC', 'Foreign FD', 'Foreign OT'
]
OWNERSHIP_CHG_COLS = [f"{col}_chg" for col in OWNERSHIP_COLS]
OWNERSHIP_CHG_RP_COLS = [f"{col}_chg_Rp" for col in OWNERSHIP_COLS]

SMART_MONEY_COLS = [
    'Foreign IS_chg_Rp', 'Foreign IB_chg_Rp', 'Foreign PF_chg_Rp', 
    'Local IS_chg_Rp', 'Local PF_chg_Rp', 'Local MF_chg_Rp', 'Local IB_chg_Rp'
]
RETAIL_COLS = ['Local ID_chg_Rp']

# ==============================================================================
# 📦 3) FUNGSI MEMUAT DATA
# ==============================================================================
def get_gdrive_service():
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None
    except KeyError:
        return None, "❌ Key 'gcp_service_account' missing in secrets."
    except Exception as e:
        return None, f"❌ Auth Error: {e}"

@st.cache_data(ttl=3600, show_spinner="🔄 Memuat data...")
def load_data():
    service, error_msg = get_gdrive_service()
    if error_msg: return pd.DataFrame(), error_msg, "error"

    try:
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1).execute()
        items = results.get('files', [])

        if not items: return pd.DataFrame(), f"❌ File '{FILE_NAME}' not found.", "error"

        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: status, done = downloader.next_chunk()
        fh.seek(0)

        df = pd.read_csv(fh, dtype=object)
        required_cols = ['Date', 'Code']
        if any(col not in df.columns for col in required_cols): return pd.DataFrame(), "❌ Missing columns.", "error"

        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'].dt.year >= 2025].copy()
        
        if df.empty: return pd.DataFrame(), "❌ No data >= 2025 found.", "error"

        df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others') if 'Sector' in df.columns else 'Others'

        cols_to_numeric = ['Price', 'Price_Chg %', 'Free Float', 'Total_Local', 'Total_Foreign', 'Top_Buyer_Vol', 'Top_Seller_Vol', 'Sec. Num', 'Top_Buyer_Value_Rp', 'Top_Seller_Value_Rp'] + OWNERSHIP_COLS + OWNERSHIP_CHG_COLS + OWNERSHIP_CHG_RP_COLS

        for col in cols_to_numeric:
            if col in df.columns:
                cleaned = df[col].astype(str).str.strip().str.replace(',', '').str.replace('[^\d.-]', '', regex=True)
                df[col] = pd.to_numeric(cleaned, errors='coerce').fillna(0)

        df = df.dropna(subset=['Date', 'Code'])
        if df.duplicated(subset=['Date', 'Code']).any():
            df = df.drop_duplicates(subset=['Date', 'Code'], keep='last')

        local_cols = [c for c in OWNERSHIP_CHG_RP_COLS if 'Local' in c and c in df.columns]
        foreign_cols = [c for c in OWNERSHIP_CHG_RP_COLS if 'Foreign' in c and c in df.columns]
        df['Total_Local_chg_Rp'] = df[local_cols].sum(axis=1)
        df['Total_Foreign_chg_Rp'] = df[foreign_cols].sum(axis=1)
        df['Total_chg_Rp'] = df['Total_Local_chg_Rp'] + df['Total_Foreign_chg_Rp']

        return df, "✅ Data Loaded", "success"

    except Exception as e:
        return pd.DataFrame(), f"❌ Error: {e}", "error"

# ==============================================================================
# 🛠️ 4) HELPER FUNCTIONS
# ==============================================================================

def format_id_short(value, is_currency=False):
    if pd.isna(value) or value == 0: return "0"
    val_abs = abs(value)
    suffix, divisor = "", 1
    if val_abs >= 1e12: suffix, divisor = " T", 1e12
    elif val_abs >= 1e9: suffix, divisor = " M", 1e9
    elif val_abs >= 1e6: suffix, divisor = " Jt", 1e6
    else: return f"{value:,.0f}"
    
    formatted = f"{value/divisor:.1f}".replace(".0", "")
    prefix = "Rp " if is_currency else ""
    return f"{prefix}{formatted}{suffix}"

def update_plotly_layout(fig):
    """
    Apply KSEI Style Clean Theme to charts.
    Forces colors to be dark to prevent 'invisible text' issues.
    """
    fig.update_layout(
        template='plotly_white', # 🔥 Kunci agar chart base-nya putih
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2B3674', family='DM Sans'), # Dark Navy Text
        xaxis=dict(
            gridcolor='#E0E5F2', 
            showgrid=True,
            tickfont=dict(color='#A3AED0') # Warna text sumbu X abu2 soft
        ), 
        yaxis=dict(
            gridcolor='#E0E5F2', 
            showgrid=True,
            tickfont=dict(color='#A3AED0')
        ),
        margin=dict(t=40, l=10, r=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

# ==============================================================================
# 🎯 5) INSTITUTIONAL INTELLIGENCE ENGINE
# ==============================================================================

# --- CROSS-SHAREHOLDING PATTERN DETECTION ---
def detect_coordinated_accumulation(df, stock_code, min_institutions=3, threshold_rp=5e9):
    """🔍 Detect if multiple institutions are accumulating simultaneously"""
    df_stock = df[df['Code'] == stock_code].sort_values('Date')
    if len(df_stock) < 3: return False, {}
    
    latest = df_stock.iloc[-1]
    # Identifikasi institusi yang sedang net buy
    institutions_accumulating = []
    for col in ['Foreign IS_chg_Rp', 'Foreign IB_chg_Rp', 'Foreign PF_chg_Rp',
                'Local IS_chg_Rp', 'Local PF_chg_Rp', 'Local MF_chg_Rp', 'Local IB_chg_Rp']:
        if col in latest and latest[col] > threshold_rp:
            inst_name = col.replace('_chg_Rp', '').replace('_', ' ')
            percentage = (latest[col] / latest.get('Sec. Num', 1)) * 100 if latest.get('Sec. Num', 1) > 0 else 0
            institutions_accumulating.append({
                'institution': inst_name,
                'amount': latest[col],
                'amount_formatted': format_id_short(latest[col], True),
                'percentage': percentage
            })
    
    is_coordinated = len(institutions_accumulating) >= min_institutions
    return is_coordinated, institutions_accumulating

# --- STEALTH ACCUMULATION DETECTOR ---
def detect_stealth_accumulation(df, stock_code, window=20):
    """🕵️ Deteksi akumulasi diam-diam (small consistent buying over time)"""
    df_stock = df[df['Code'] == stock_code].sort_values('Date')
    if len(df_stock) < window: return False, None
    
    # Hitung cumulative flow tanpa spike besar
    flow_cols = [c for c in OWNERSHIP_CHG_RP_COLS if any(x in c for x in ['Foreign IS', 'Foreign IB', 'Foreign PF', 
                                                                          'Local IS', 'Local IB', 'Local PF', 'Local MF'])]
    df_stock['Stealth_Flow'] = df_stock[flow_cols].sum(axis=1)
    
    # Cari pattern: flow positif konsisten tanpa volume spike
    df_window = df_stock.tail(window)
    positive_days = (df_window['Stealth_Flow'] > 0).sum()
    avg_flow = df_window['Stealth_Flow'].mean()
    max_flow = df_window['Stealth_Flow'].max()
    
    # Criteria: 70% hari positif, rata2 flow moderat, tidak ada spike ekstrem
    is_stealth = (positive_days/window >= 0.7) and (avg_flow > 1e9) and (max_flow < 10e9)
    
    stealth_score = min(100, (positive_days/window * 50) + (min(avg_flow/1e9, 5) * 10))
    
    return is_stealth, {
        'positive_days': positive_days,
        'avg_daily_flow': avg_flow,
        'total_stealth_accumulation': df_window['Stealth_Flow'].sum(),
        'max_single_day_flow': max_flow,
        'stealth_score': stealth_score
    }

# --- INSTITUTIONAL CONVICTION SCORE ---
def calculate_institutional_conviction(df, stock_code):
    """📊 Skor keyakinan institusional (0-100)"""
    df_stock = df[df['Code'] == stock_code].sort_values('Date')
    if len(df_stock) < 5: return 0, {}
    
    latest = df_stock.iloc[-1]
    score_components = {}
    
    # 1. Net Institutional Flow (30 points)
    inst_cols = [c for c in OWNERSHIP_CHG_RP_COLS if any(x in c for x in ['IS', 'IB', 'PF', 'MF'])]
    net_inst_flow = sum(latest.get(c, 0) for c in inst_cols)
    max_inst_flow = df_stock[inst_cols].sum(axis=1).abs().max() if len(df_stock) > 0 else 1
    score_flow = min(30, (abs(net_inst_flow) / max(max_inst_flow, 1)) * 30)
    score_components['institutional_flow'] = score_flow
    
    # 2. Flow Consistency (25 points)
    recent_flows = df_stock.tail(5)[inst_cols].sum(axis=1) if len(df_stock) >= 5 else pd.Series([0])
    consistency = (recent_flows > 0).sum() / len(recent_flows) if net_inst_flow > 0 else (recent_flows < 0).sum() / len(recent_flows)
    score_components['consistency'] = consistency * 25
    
    # 3. Ownership Concentration (20 points)
    inst_ownership = sum(latest.get(c.replace('_chg_Rp', ''), 0) for c in inst_cols)
    total_shares = latest.get('Sec. Num', 1)
    concentration = (inst_ownership / total_shares) * 100 if total_shares > 0 else 0
    score_components['concentration'] = min(20, concentration)
    
    # 4. Price vs Flow Divergence (15 points)
    price_change = latest.get('Price_Chg %', 0)
    
    if net_inst_flow > 0 and price_change < 0:  # Akumulasi saat harga turun
        score_components['divergence'] = 15
    elif net_inst_flow < 0 and price_change > 0:  # Distribusi saat harga naik
        score_components['divergence'] = 15
    else:
        score_components['divergence'] = 0
    
    # 5. Recent Acceleration (10 points)
    if len(df_stock) >= 10:
        last_5_flow = df_stock.tail(5)[inst_cols].sum(axis=1).sum()
        prev_5_flow = df_stock.iloc[-10:-5][inst_cols].sum(axis=1).sum()
        acceleration = abs(last_5_flow) - abs(prev_5_flow)
        score_components['acceleration'] = min(10, max(0, acceleration / 1e9))
    else:
        score_components['acceleration'] = 0
    
    total_score = sum(score_components.values())
    return min(100, total_score), score_components

# --- SMART MONEY CLUSTERING ANALYSIS ---
def cluster_smart_money_patterns(df, n_clusters=4, sample_size=200):
    """🤖 Cluster saham berdasarkan pattern smart money"""
    # Siapkan features
    features = []
    stock_codes = []
    sectors = []
    
    for code in df['Code'].unique()[:sample_size]:  # Batasi untuk performa
        df_stock = df[df['Code'] == code].tail(20)  # 20 hari terakhir
        if len(df_stock) < 5: continue
        
        # Features: flow pattern, volatility, accumulation rate
        flow_cols = [c for c in SMART_MONEY_COLS if c in df_stock.columns]
        retail_cols = [c for c in RETAIL_COLS if c in df_stock.columns]
        
        smart_flow = df_stock[flow_cols].sum().sum() if flow_cols else 0
        retail_flow = df_stock[retail_cols].sum().sum() if retail_cols else 0
        flow_ratio = smart_flow / abs(retail_flow) if retail_flow != 0 else 10
        
        price_volatility = df_stock['Price'].pct_change().std() * 100 if len(df_stock) > 1 else 0
        accumulation_days = (df_stock[flow_cols].sum(axis=1) > 0).sum() if flow_cols else 0
        
        features.append([
            smart_flow / 1e9,  # Smart money flow (miliar)
            flow_ratio,        # Smart vs retail ratio
            price_volatility,  # Volatilitas harga
            accumulation_days, # Hari akumulasi
        ])
        stock_codes.append(code)
        sectors.append(df_stock.iloc[-1].get('Sector', 'N/A'))
    
    if len(features) < n_clusters: return pd.DataFrame()
    
    # Clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Interpretasi cluster
    cluster_labels = {
        0: "🚀 Strong Accumulation",
        1: "🕵️ Stealth Accumulation", 
        2: "⚠️ Distribution",
        3: "📊 Sideways Accumulation",
        4: "🎯 High Conviction" if n_clusters > 4 else "Other"
    }
    
    results = pd.DataFrame({
        'Code': stock_codes,
        'Sector': sectors,
        'Cluster': clusters,
        'Cluster_Label': [cluster_labels.get(c, f"Cluster {c}") for c in clusters],
        'Smart_Flow_Miliar': [f[0] for f in features],
        'Flow_Ratio': [f[1] for f in features],
        'Volatility': [f[2] for f in features]
    })
    
    return results.sort_values('Smart_Flow_Miliar', ascending=False)

# --- INSTITUTIONAL FOOTPRINT TRACKER ---
def track_institutional_footprint(df, window_days=60):
    """🗺️ Track institutional footprint changes over time"""
    latest_date = df['Date'].max()
    start_date = latest_date - pd.Timedelta(days=window_days)
    
    df_window = df[df['Date'] >= start_date]
    
    results = []
    for code in df_window['Code'].unique()[:200]:  # Batasi untuk performa
        df_stock = df_window[df_window['Code'] == code]
        if len(df_stock) < 2: continue
        
        # Calculate institutional footprint change
        inst_cols = [c for c in OWNERSHIP_COLS if any(x in c for x in ['IS', 'IB', 'PF', 'MF'])]
        
        start_ownership = df_stock.iloc[0][inst_cols].sum()
        end_ownership = df_stock.iloc[-1][inst_cols].sum()
        ownership_change = end_ownership - start_ownership
        
        # Calculate flow momentum
        flow_cols = [f"{c}_chg_Rp" for c in inst_cols if f"{c}_chg_Rp" in df_stock.columns]
        total_inst_flow = df_stock[flow_cols].sum().sum()
        
        if abs(total_inst_flow) > 1e9:  # Minimal 1 miliar
            flow_percentage = (ownership_change / df_stock.iloc[-1].get('Sec. Num', 1)) * 100 if df_stock.iloc[-1].get('Sec. Num', 1) > 0 else 0
            
            results.append({
                'Code': code,
                'Sector': df_stock.iloc[-1].get('Sector', 'N/A'),
                'Price': df_stock.iloc[-1].get('Price', 0),
                'Ownership_Change': ownership_change,
                'Total_Inst_Flow': total_inst_flow,
                'Flow_Percentage': flow_percentage,
                'Footprint_Score': min(100, abs(total_inst_flow) / 1e9 * 10)  # Skor 0-100
            })
    
    return pd.DataFrame(results).sort_values('Footprint_Score', ascending=False)

# --- HIGH CONVICTION SCANNER ---
@st.cache_data
def scan_high_conviction_stocks(df, min_score=75, min_flow=10e9):
    """🏆 Scan untuk saham dengan conviction tinggi"""
    results = []
    
    for code in df['Code'].unique()[:100]:  # Batasi untuk performa
        df_stock = df[df['Code'] == code].sort_values('Date')
        if len(df_stock) < 10: continue
        
        latest = df_stock.iloc[-1]
        
        # Hitung institutional flow
        inst_cols = [c for c in OWNERSHIP_CHG_RP_COLS if any(x in c for x in ['IS', 'IB', 'PF', 'MF'])]
        inst_flow = sum(latest.get(c, 0) for c in inst_cols)
        
        if abs(inst_flow) < min_flow: continue
        
        # Hitung conviction score
        score, components = calculate_institutional_conviction(df, code)
        
        if score >= min_score:
            is_stealth, stealth_details = detect_stealth_accumulation(df, code)
            is_coordinated, coord_details = detect_coordinated_accumulation(df, code, min_institutions=2)
            
            results.append({
                'Code': code,
                'Sector': latest.get('Sector', 'N/A'),
                'Price': latest.get('Price', 0),
                'Price_Chg_%': latest.get('Price_Chg %', 0),
                'Conviction_Score': score,
                'Institutional_Flow': inst_flow,
                'Is_Stealth': is_stealth,
                'Is_Coordinated': is_coordinated,
                'Stealth_Score': stealth_details.get('stealth_score', 0) if is_stealth else 0,
                'Coordinated_Count': len(coord_details) if is_coordinated else 0
            })
    
    return pd.DataFrame(results).sort_values('Conviction_Score', ascending=False)

# ==============================================================================
# 📊 6) EXISTING HELPER FUNCTIONS (DARI SEBELUMNYA)
# ==============================================================================

@st.cache_data
def calculate_macro_flow(df_filtered):
    net_flow = df_filtered[OWNERSHIP_CHG_RP_COLS].sum().reset_index()
    net_flow.columns = ['Kategori', 'Total Net Flow (Rp)']
    net_flow['Kategori'] = net_flow['Kategori'].str.replace('_chg_Rp', '')
    net_flow = net_flow.sort_values(by='Total Net Flow (Rp)', ascending=False)
    
    cum_flow = df_filtered.groupby('Date')[['Total_Local_chg_Rp', 'Total_Foreign_chg_Rp']].sum().cumsum().reset_index()
    cum_flow = cum_flow.melt('Date', var_name='Kategori', value_name='Cumulative Flow (Rp)')
    cum_flow['Kategori'] = cum_flow['Kategori'].str.replace('_chg_Rp', ' (Net Rp)')
    return net_flow, cum_flow

@st.cache_data
def calculate_sector_rotation(df_filtered, selected_category):
    col = f"{selected_category}_chg_Rp"
    if col not in df_filtered.columns: return pd.DataFrame(), "Col missing"
    sector_flow = df_filtered.groupby('Sector')[col].sum().reset_index()
    sector_flow.columns = ['Sector', 'Net Flow (Rp)']
    return sector_flow.sort_values('Net Flow (Rp)', ascending=False), None

@st.cache_data
def get_stock_ownership_state(df, stock_code):
    df_stock = df[df['Code'] == stock_code]
    if df_stock.empty: return pd.DataFrame(), pd.Series(dtype='object')
    latest = df_stock.sort_values('Date').iloc[-1]
    df_state = latest[OWNERSHIP_COLS].reset_index()
    df_state.columns = ['Kategori', 'Jumlah Saham']
    return df_state, latest

@st.cache_data
def calculate_monthly_change_table(df_stock):
    df_res = df_stock.sort_values('Date', ascending=False)[['Date'] + OWNERSHIP_CHG_COLS].copy()
    df_res.rename(columns={'Date': 'Bulan'}, inplace=True)
    df_res['Bulan'] = df_res['Bulan'].dt.strftime('%b %Y')
    return df_res

@st.cache_data
def calculate_smart_money_signals(df_year, window_periods=3, min_acc_threshold=5e9):
    if df_year.empty: return pd.DataFrame()
    results = []
    for code in df_year['Code'].unique():
        df_w = df_year[df_year['Code'] == code].sort_values('Date').tail(window_periods)
        if df_w.empty: continue
        
        last_p = df_w.iloc[-1]['Price']
        start_p = df_w.iloc[0]['Price']
        pct = ((last_p - start_p)/start_p)*100 if start_p > 0 else 0
        
        sm_sum = df_w[[c for c in SMART_MONEY_COLS if c in df_w]].sum().sum()
        ret_sum = df_w[[c for c in RETAIL_COLS if c in df_w]].sum().sum()
        
        status, score = "Netral", 0
        if sm_sum > min_acc_threshold and ret_sum < 0: status, score = "🔥 Big Accumulation", 100
        elif sm_sum > (min_acc_threshold/2) and pct <= 3: status, score = "💎 Divergence", 80
        elif sm_sum < -min_acc_threshold and ret_sum > 0: status, score = "⚠️ Distribution", -50
        
        if abs(score) >= 50:
            results.append({'Code': code, 'Sector': df_w.iloc[-1].get('Sector','N/A'), 'Price': last_p, 
                           'Price Chg %': pct, 'Smart Money (Rp)': sm_sum, 'Retail (Rp)': ret_sum, 'Signal': status})
    
    df_res = pd.DataFrame(results)
    if not df_res.empty: df_res = df_res.sort_values('Smart Money (Rp)', ascending=False)
    return df_res

@st.cache_data
def get_significant_movements(df_month, threshold_rp=10e9, threshold_pct=5):
    if df_month.empty: return pd.DataFrame()
    results = []
    available_rp_cols = [c for c in OWNERSHIP_CHG_RP_COLS if c in df_month.columns]

    for code in df_month['Code'].unique():
        row = df_month[df_month['Code'] == code].iloc[0]
        abs_flow = sum(abs(row[c]) for c in available_rp_cols)
        net_flow = row.get('Total_chg_Rp', 0)
        shares = row.get('Sec. Num', 1)
        pct = (abs_flow / shares * 100) if shares > 0 else 0
        
        if abs_flow >= threshold_rp or pct >= threshold_pct:
            direction = "NET BUY" if net_flow > 0 else "NET SELL" if net_flow < 0 else "NEUTRAL"
            
            vals = {c.replace('_chg_Rp', ''): row[c] for c in available_rp_cols}
            top_b_cat = max(vals, key=vals.get) if vals else "-"
            top_s_cat = min(vals, key=vals.get) if vals else "-"
            
            buyer_str = f"{top_b_cat}" if vals[top_b_cat] > 0 else "-"
            seller_str = f"{top_s_cat}" if vals[top_s_cat] < 0 else "-"

            results.append({
                'Code': code, 'Sector': row.get('Sector','N/A'), 'Price': row.get('Price',0),
                'Total Flow (Rp)': abs_flow, 'Net Flow (Rp)': net_flow, 'Flow %': pct, 
                'Direction': direction, 'Top_Buyer': buyer_str, 'Top_Seller': seller_str
            })
    
    df_res = pd.DataFrame(results)
    if not df_res.empty: df_res = df_res.sort_values('Total Flow (Rp)', ascending=False)
    return df_res

# --- SANKEY CHART ---
def create_sankey_chart(df, stock_code, selected_date, mode='Volume'):
    row = df[(df['Code'] == stock_code) & (df['Date'] == selected_date)]
    if row.empty: return None
    row = row.iloc[0]
    
    cols = OWNERSHIP_CHG_COLS if mode == 'Volume' else OWNERSHIP_CHG_RP_COLS
    is_rp = (mode == 'Value')
    
    sellers, buyers, total_vol = [], [], 0
    for col in cols:
        val = row[col] if col in row else 0
        cat = col.replace('_chg_Rp', '').replace('_chg', '')
        if val != 0:
            lbl = f"{cat}\n({format_id_short(abs(val), is_rp)})"
            if val < 0: sellers.append({'l': lbl, 'v': abs(val)}); total_vol += abs(val)
            else: buyers.append({'l': lbl, 'v': val})
            
    if not sellers and not buyers: return None
    
    labels = [f"MARKET\n({format_id_short(total_vol, is_rp)})"] + [s['l'] for s in sellers] + [b['l'] for b in buyers]
    colors = ["#E0E5F2"] + ["#EE5D50"]*len(sellers) + ["#05CD99"]*len(buyers) 
    
    source = list(range(1, len(sellers)+1)) + [0]*len(buyers)
    target = [0]*len(sellers) + list(range(len(sellers)+1, len(labels)))
    values = [s['v'] for s in sellers] + [b['v'] for b in buyers]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=20, thickness=15, line=dict(color="white", width=0.5), label=labels, color=colors),
        link=dict(source=source, target=target, value=values, color=['rgba(238, 93, 80, 0.4)']*len(sellers) + ['rgba(5, 205, 153, 0.4)']*len(buyers))
    )])
    fig.update_layout(title_text=f"Arus Dana {stock_code} ({selected_date.strftime('%b %Y')}) - {mode}", font_size=14, height=500)
    return update_plotly_layout(fig)

# ==============================================================================
# 💎 7) LAYOUT UTAMA & SIDEBAR
# ==============================================================================

# --- SIDEBAR (Updated Look) ---
with st.sidebar:
    # Logo Placeholder
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=50)
    st.markdown("<h3 style='color:#2B3674; margin-top:0;'>Bandarmology PRO</h3>", unsafe_allow_html=True)
    st.divider()

    st.markdown("##### Global Controls")
    if st.button("🔄 Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Load Data Block
    df, status_msg, status_level = load_data()
    if status_level == "error":
        st.error(status_msg)
        st.stop()
    
    # Month Filter
    available_months = sorted(df['Date'].dt.strftime('%Y-%m').unique(), reverse=True)
    selected_month_str = st.selectbox("📅 Pilih Bulan", available_months)
    selected_month = pd.to_datetime(selected_month_str + '-01')
    df_filtered_month = df[df['Date'].dt.strftime('%Y-%m') == selected_month_str]

    st.divider()
    with st.expander("⚙️ Advanced Filter"):
        threshold_rp = st.number_input("Min Flow (Rp)", value=10_000_000_000, step=1_000_000_000, format="%d")
        min_rotation = st.number_input("Min Rotation (Rp)", value=1_000_000_000, step=500_000_000, format="%d")
        conviction_threshold = st.slider("Min Conviction Score", 0, 100, 75)

# --- MAIN PAGE HEADER (Purple Gradient) ---
st.markdown(f"""
    <div class="header-banner">
        <div class="header-title">Welcome, Institutional Trader! 👋</div>
        <div class="header-subtitle">Advanced Bandarmology & Institutional Intelligence Engine - Periode <b>{selected_month.strftime('%B %Y')}</b></div>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 📑 TABS VISUALISASI (DENGAN TAB BARU INSTITUTIONAL INTELLIGENCE)
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌍 Market Overview", "🏭 Sector Rotation", "📈 Deep Dive", "🔍 Big Screener", 
    "🤖 Smart Signals", "🔥 Top Movers", "🎯 Institutional Intel"
])

# --- TAB 1: MAKRO ---
with tab1:
    col_m1, col_m2 = st.columns([2, 1])
    
    df_net, df_cum = calculate_macro_flow(df)
    
    with col_m1:
        st.markdown('<div class="css-card"><div class="card-title">📊 Cumulative Flow YTD (Rp)</div>', unsafe_allow_html=True)
        fig_macro = px.line(df_cum, x='Date', y='Cumulative Flow (Rp)', color='Kategori', 
                           color_discrete_map={'Total_Local (Net Rp)': '#05CD99', 'Total_Foreign (Net Rp)': '#EE5D50'})
        st.plotly_chart(update_plotly_layout(fig_macro), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m2:
        st.markdown('<div class="css-card"><div class="card-title">🏆 Top Net Buy (YTD)</div>', unsafe_allow_html=True)
        fig_buy = px.bar(df_net.head(7), x='Total Net Flow (Rp)', y='Kategori', orientation='h', color_discrete_sequence=['#05CD99'])
        fig_buy.update_layout(height=300, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(update_plotly_layout(fig_buy), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="css-card"><div class="card-title">🔻 Top Net Sell (YTD)</div>', unsafe_allow_html=True)
        fig_sell = px.bar(df_net.tail(7), x='Total Net Flow (Rp)', y='Kategori', orientation='h', color_discrete_sequence=['#EE5D50'])
        fig_sell.update_layout(height=300, yaxis={'categoryorder':'total descending'})
        st.plotly_chart(update_plotly_layout(fig_sell), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: SEKTOR ---
with tab2:
    col_sel, col_chart = st.columns([1, 4])
    with col_sel:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Filter</div>', unsafe_allow_html=True)
        cats = sorted([c.replace('_chg_Rp','') for c in OWNERSHIP_CHG_RP_COLS])
        sel_cat = st.selectbox("Pilih Investor:", cats, index=cats.index('Foreign IB') if 'Foreign IB' in cats else 0)
        df_sec, _ = calculate_sector_rotation(df_filtered_month, sel_cat)
        st.metric("Total Flow", f"Rp {format_id_short(df_sec['Net Flow (Rp)'].sum())}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_chart:
        if not df_sec.empty:
            st.markdown(f'<div class="css-card"><div class="card-title">Rotasi Sektor {sel_cat} ({selected_month_str})</div>', unsafe_allow_html=True)
            fig_sec = px.bar(df_sec, x='Net Flow (Rp)', y='Sector', orientation='h', 
                             color='Net Flow (Rp)', color_continuous_scale='RdYlGn')
            fig_sec.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(update_plotly_layout(fig_sec), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Data tidak tersedia.")

# --- TAB 3: INDIVIDUAL (UPDATED LAYOUT: SANKEY TOP, HIST BOTTOM) ---
with tab3:
    stocks_avail = sorted(df['Code'].unique())
    sel_stock = st.selectbox("🔎 Cari Kode Saham:", stocks_avail, index=stocks_avail.index('BBRI') if 'BBRI' in stocks_avail else 0)
    
    # Data Prep
    df_stock_all = df[df['Code'] == sel_stock].sort_values('Date')
    df_stock_month = df_stock_all[df_stock_all['Date'].dt.strftime('%Y-%m') == selected_month_str]
    
    if df_stock_month.empty:
        st.warning(f"Data {sel_stock} bulan {selected_month_str} kosong. Menggunakan data terakhir tersedia.")
        df_state, last_row = get_stock_ownership_state(df_stock_all, sel_stock)
    else:
        df_state, last_row = get_stock_ownership_state(df_stock_month, sel_stock)

    # 1. KPI Cards + Institutional Metrics
    k1, k2, k3, k4 = st.columns(4)
    
    # Price
    k1.markdown(f"""<div class="css-card" style="text-align:center;">
        <div style="font-size:14px; color:#A3AED0;">Harga Terakhir</div>
        <div style="font-size:24px; font-weight:700; color:#2B3674;">Rp {last_row['Price']:,.0f}</div>
    </div>""", unsafe_allow_html=True)
    
    # Flow dengan warna
    flow_val = last_row.get('Total_chg_Rp',0)
    flow_color = '#05CD99' if flow_val > 0 else '#EE5D50'
    k2.markdown(f"""<div class="css-card" style="text-align:center;">
        <div style="font-size:14px; color:#A3AED0;">Flow {selected_month_str}</div>
        <div style="font-size:24px; font-weight:700; color: {flow_color};">
            {format_id_short(flow_val, True)}
        </div>
    </div>""", unsafe_allow_html=True)
    
    # Conviction Score
    conviction_score, _ = calculate_institutional_conviction(df, sel_stock)
    badge_color = "badge-high" if conviction_score >= 80 else "badge-medium" if conviction_score >= 60 else "badge-low"
    k3.markdown(f"""<div class="css-card" style="text-align:center;">
        <div style="font-size:14px; color:#A3AED0;">Conviction Score</div>
        <div style="font-size:24px; font-weight:700; color:#2B3674;">{conviction_score:.0f}</div>
        <div><span class="{badge_color}">{
            'HIGH' if conviction_score >= 80 else 'MEDIUM' if conviction_score >= 60 else 'LOW'
        }</span></div>
    </div>""", unsafe_allow_html=True)
    
    # Stealth Detection
    is_stealth, stealth_details = detect_stealth_accumulation(df, sel_stock)
    stealth_badge = "🕵️" if is_stealth else ""
    k4.markdown(f"""<div class="css-card" style="text-align:center;">
        <div style="font-size:14px; color:#A3AED0;">Pattern Detection</div>
        <div style="font-size:18px; font-weight:700; color:#2B3674;">{last_row.get('Sector','-')}</div>
        <div style="font-size:12px; margin-top:5px;">
            {stealth_badge} {'Stealth Detected' if is_stealth else 'Normal Pattern'}
        </div>
    </div>""", unsafe_allow_html=True)

    # 2. SANKEY CHART (Full Width)
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    mode_sankey = st.radio("Mode Visualisasi:", ["Value (Rp)", "Volume (Lot)"], horizontal=True, label_visibility="collapsed")
    mode_key = 'Value' if 'Rp' in mode_sankey else 'Volume'
    
    date_for_sankey = df_stock_month.iloc[0]['Date'] if not df_stock_month.empty else last_row['Date']
    fig_sankey = create_sankey_chart(df, sel_stock, date_for_sankey, mode=mode_key)
    
    if fig_sankey:
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("Pergerakan tidak cukup signifikan untuk Sankey.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. INSTITUTIONAL INTELLIGENCE PANEL
    st.markdown('<div class="css-card"><div class="card-title">🎯 Institutional Intelligence</div>', unsafe_allow_html=True)
    
    col_inst1, col_inst2 = st.columns(2)
    
    with col_inst1:
        # Coordinated Accumulation Check
        st.markdown("#### 🤝 Coordinated Accumulation")
        is_coordinated, institutions = detect_coordinated_accumulation(df, sel_stock)
        
        if is_coordinated:
            st.success(f"✅ **COORDINATED ACCUMULATION DETECTED!**")
            st.markdown(f"**{len(institutions)} institutions accumulating:**")
            for inst in institutions:
                st.markdown(f"• **{inst['institution']}**: {inst['amount_formatted']} ({inst['percentage']:.2f}%)")
        else:
            st.info("No coordinated accumulation detected.")
    
    with col_inst2:
        # Stealth Accumulation Details
        st.markdown("#### 🕵️ Stealth Accumulation Analysis")
        if is_stealth:
            st.success(f"✅ **STEALTH ACCUMULATION DETECTED!**")
            st.markdown(f"""
            - **Score**: {stealth_details['stealth_score']:.0f}/100
            - **Positive Days**: {stealth_details['positive_days']}/20
            - **Total Accumulation**: {format_id_short(stealth_details['total_stealth_accumulation'], True)}
            - **Max Single Day**: {format_id_short(stealth_details['max_single_day_flow'], True)}
            """)
        else:
            st.info("No stealth pattern detected.")
    
    # Conviction Score Breakdown
    st.markdown("#### 📊 Conviction Score Breakdown")
    _, score_components = calculate_institutional_conviction(df, sel_stock)
    
    fig_conviction = go.Figure(data=[
        go.Bar(name='Score Components', 
               x=list(score_components.keys()),
               y=list(score_components.values()),
               marker_color=['#4318FF', '#05CD99', '#FFB547', '#868CFF', '#EE5D50'])
    ])
    fig_conviction.update_layout(height=300, showlegend=False)
    st.plotly_chart(update_plotly_layout(fig_conviction), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. MONTHLY HISTORY (Full Width - Below Sankey)
    st.markdown('<div class="css-card"><div class="card-title">📅 Monthly History (Deep Trends)</div>', unsafe_allow_html=True)
    df_hist = calculate_monthly_change_table(df_stock_all)
    
    try:
        st.dataframe(
            df_hist.style.format("{:,.0f}", subset=OWNERSHIP_CHG_COLS)
            .background_gradient(cmap='RdYlGn', subset=OWNERSHIP_CHG_COLS, axis=1),
            use_container_width=True, hide_index=True
        )
    except:
        st.dataframe(
            df_hist.style.format("{:,.0f}", subset=OWNERSHIP_CHG_COLS),
            use_container_width=True, hide_index=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: SCREENER ---
with tab4:
    st.markdown(f'<div class="css-card"><div class="card-title">🔍 Big Rotation Screener ({selected_month_str})</div>', unsafe_allow_html=True)
    
    df_scr = df_filtered_month.copy()
    if not df_scr.empty:
        mask = (df_scr['Top_Buyer_Value_Rp'].abs() >= min_rotation) | (df_scr['Top_Seller_Value_Rp'].abs() >= min_rotation)
        df_scr = df_scr[mask].sort_values('Top_Buyer_Value_Rp', ascending=False)
        
        if not df_scr.empty:
            # Tambahkan Institutional Metrics
            df_scr['Conviction_Score'] = df_scr['Code'].apply(lambda x: calculate_institutional_conviction(df, x)[0])
            df_scr['Is_Stealth'] = df_scr['Code'].apply(lambda x: detect_stealth_accumulation(df, x)[0])
            df_scr['Is_Coordinated'] = df_scr['Code'].apply(lambda x: detect_coordinated_accumulation(df, x, min_institutions=2)[0])
            
            disp = df_scr[['Code', 'Sector', 'Conviction_Score', 'Is_Stealth', 'Is_Coordinated', 
                          'Top_Buyer', 'Top_Buyer_Value_Rp', 'Top_Seller', 'Top_Seller_Value_Rp', 'Total_chg_Rp']]
            disp.columns = ['Code', 'Sector', 'Conviction', 'Stealth', 'Coordinated', 
                           'Buyer', 'Buy Val', 'Seller', 'Sell Val', 'Net Flow']
            
            # Style dengan institutional flags
            def style_conviction(val):
                color = '#0D9D58' if val >= 80 else '#FF9800' if val >= 60 else '#FF3B30'
                return f'color: {color}; font-weight: bold;'
            
            def style_flag(val):
                if val:
                    return 'background-color: #D6F5E3; color: #0D9D58; font-weight: bold; text-align: center;'
                return ''
            
            styled_df = disp.style\
                .format({'Buy Val': '{:,.0f}', 'Sell Val': '{:,.0f}', 'Net Flow': '{:,.0f}', 'Conviction': '{:.0f}'})\
                .applymap(style_conviction, subset=['Conviction'])\
                .applymap(style_flag, subset=['Stealth', 'Coordinated'])\
                .bar(subset=['Net Flow'], align='mid', color=['#EE5D50', '#05CD99'])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)
        else:
            st.info(f"Tidak ada rotasi > Rp {min_rotation:,.0f}")
    else:
        st.warning("Data bulan ini kosong.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: AI SIGNALS ---
with tab5:
    col_ai_filt, col_ai_res = st.columns([1, 3])
    with col_ai_filt:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Parameters</div>', unsafe_allow_html=True)
        win = st.slider("Window (Bulan)", 1, 6, 3)
        min_acc = st.number_input("Min Akumulasi (Miliar)", 5.0, 100.0, 5.0) * 1e9
        sec_filter = st.selectbox("Sektor", ["All"] + sorted(df['Sector'].unique().tolist()))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_ai_res:
        date_cutoff = df['Date'].max() - pd.DateOffset(months=6)
        df_sig = calculate_smart_money_signals(df[df['Date'] >= date_cutoff], win, min_acc)
        
        if not df_sig.empty:
            if sec_filter != "All": df_sig = df_sig[df_sig['Sector'] == sec_filter]
            
            st.markdown(f'<div class="css-card"><div class="card-title">💎 Smart Money Signals ({len(df_sig)})</div>', unsafe_allow_html=True)
            
            def color_sig(val):
                c = '#B78202' if 'Accumulation' in val else '#027A9B' if 'Divergence' in val else '#A61021'
                bg = '#FFF8E1' if 'Accumulation' in val else '#E1F5FE' if 'Divergence' in val else '#FFEBEE'
                return f'color: {c}; background-color: {bg}; font-weight: 700;'

            st.dataframe(
                df_sig.style.format({'Price': '{:,.0f}', 'Price Chg %': '{:.2f}%', 'Smart Money (Rp)': '{:,.0f}', 'Retail (Rp)': '{:,.0f}'})
                .applymap(color_sig, subset=['Signal']),
                use_container_width=True, hide_index=True
            )
            
            # Scatter Plot
            fig_scat = px.scatter(df_sig, x='Smart Money (Rp)', y='Price Chg %', color='Signal', 
                                  size='Price', hover_data=['Code'], text='Code',
                                  color_discrete_map={'🔥 Big Accumulation': '#FFB547', '💎 Divergence': '#4318FF', '⚠️ Distribution': '#EE5D50'})
            st.plotly_chart(update_plotly_layout(fig_scat), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Belum ada sinyal yang cocok.")

# --- TAB 6: HOT MOVERS ---
with tab6:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f"<div class='card-title'>Pergerakan Signifikan: {selected_month_str}</div>", unsafe_allow_html=True)
    with col_h2:
        threshold_pct_tab6 = st.number_input("Threshold % Saham", 0.1, 20.0, 5.0, 0.5)

    df_hot = get_significant_movements(df_filtered_month, threshold_rp=10e9, threshold_pct=threshold_pct_tab6)
    
    if not df_hot.empty:
        # Tambahkan Institutional Metrics
        df_hot['Conviction_Score'] = df_hot['Code'].apply(lambda x: calculate_institutional_conviction(df, x)[0])
        df_hot['Is_Stealth'] = df_hot['Code'].apply(lambda x: detect_stealth_accumulation(df, x)[0])
        
        # Top Movers Chart
        top10 = df_hot.head(10)
        fig_hot = px.bar(top10, x='Code', y='Total Flow (Rp)', color='Direction',
                         color_discrete_map={'NET BUY': '#05CD99', 'NET SELL': '#EE5D50'},
                         text='Flow %', hover_data=['Conviction_Score', 'Is_Stealth'])
        fig_hot.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_hot.update_layout(height=400)
        st.plotly_chart(update_plotly_layout(fig_hot), use_container_width=True)
        
        # Detail Table
        st.dataframe(
            df_hot[['Code', 'Sector', 'Price', 'Total Flow (Rp)', 'Direction', 'Conviction_Score', 
                   'Is_Stealth', 'Top_Buyer', 'Top_Seller']]
            .style.format("{:,.0f}", subset=['Price', 'Total Flow (Rp)'])
            .format("{:.0f}", subset=['Conviction_Score'])
            .applymap(lambda v: f'color: {"#05CD99" if v=="NET BUY" else "#EE5D50" if v=="NET SELL" else "#555"}; font-weight:bold;', subset=['Direction'])
            .applymap(lambda v: 'background-color: #E5F3FF;' if v else '', subset=['Is_Stealth']),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Tidak ada pergerakan ekstrem di atas threshold.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 🎯 TAB 7: INSTITUTIONAL INTELLIGENCE (NEW!)
# ==============================================================================
with tab7:
    st.markdown('<div class="header-banner" style="margin-bottom:20px; padding:20px;">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">🎯 Institutional Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Advanced detection of institutional accumulation patterns & high-conviction plays</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 1: HIGH CONVICTION STOCKS ---
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🏆 High Conviction Stocks (Score ≥ 75)</div>', unsafe_allow_html=True)
    
    with st.spinner("Scanning for high conviction stocks..."):
        df_high_conviction = scan_high_conviction_stocks(df, min_score=conviction_threshold, min_flow=5e9)
    
    if not df_high_conviction.empty:
        # Top Conviction Stocks Chart
        top_conviction = df_high_conviction.head(10)
        fig_conv = px.bar(top_conviction, x='Code', y='Conviction_Score', 
                         color='Is_Stealth', color_discrete_map={True: '#4318FF', False: '#05CD99'},
                         hover_data=['Sector', 'Price', 'Institutional_Flow'],
                         text='Conviction_Score')
        fig_conv.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_conv.update_layout(height=400)
        st.plotly_chart(update_plotly_layout(fig_conv), use_container_width=True)
        
        # Detail Table dengan badges
        def add_badges(row):
            badges = []
            if row['Conviction_Score'] >= 80:
                badges.append('<span class="badge badge-high">HIGH</span>')
            if row['Is_Stealth']:
                badges.append('<span class="badge badge-stealth">STEALTH</span>')
            if row['Is_Coordinated']:
                badges.append('<span class="badge badge-coordinated">COORDINATED</span>')
            return ' '.join(badges)
        
        df_display = df_high_conviction.copy()
        df_display['Badges'] = df_display.apply(add_badges, axis=1)
        df_display['Institutional_Flow'] = df_display['Institutional_Flow'].apply(lambda x: format_id_short(x, True))
        
        st.markdown("#### 📋 Detailed Analysis")
        st.dataframe(
            df_display[['Code', 'Sector', 'Price', 'Price_Chg_%', 'Conviction_Score', 
                       'Institutional_Flow', 'Badges']]
            .style.format({'Price': '{:,.0f}', 'Price_Chg_%': '{:.2f}%', 'Conviction_Score': '{:.0f}'}),
            use_container_width=True, hide_index=True
        )
    else:
        st.info(f"Tidak ada saham dengan conviction score ≥ {conviction_threshold}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 2: SMART MONEY CLUSTERING ---
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🤖 Smart Money Pattern Clustering</div>', unsafe_allow_html=True)
    
    col_clust1, col_clust2 = st.columns(2)
    with col_clust1:
        n_clusters = st.slider("Number of Clusters", 3, 6, 4)
        sample_size = st.slider("Sample Size", 50, 500, 200)
    
    with st.spinner("Clustering smart money patterns..."):
        df_clusters = cluster_smart_money_patterns(df, n_clusters=n_clusters, sample_size=sample_size)
    
    if not df_clusters.empty:
        # Cluster Visualization
        fig_cluster = px.scatter(df_clusters, x='Smart_Flow_Miliar', y='Volatility',
                                color='Cluster_Label', size='Flow_Ratio',
                                hover_data=['Code', 'Sector', 'Flow_Ratio'],
                                title="Smart Money Pattern Clusters")
        st.plotly_chart(update_plotly_layout(fig_cluster), use_container_width=True)
        
        # Cluster Summary
        st.markdown("#### 📊 Cluster Summary")
        cluster_summary = df_clusters.groupby('Cluster_Label').agg({
            'Code': 'count',
            'Smart_Flow_Miliar': 'mean',
            'Volatility': 'mean'
        }).round(2).reset_index()
        
        cluster_summary.columns = ['Cluster', 'Count', 'Avg Smart Flow (Miliar)', 'Avg Volatility']
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        for idx, row in cluster_summary.iterrows():
            if idx == 0:
                with col_sum1:
                    st.metric(f"{row['Cluster']}", f"{row['Count']} stocks")
            elif idx == 1:
                with col_sum2:
                    st.metric(f"{row['Cluster']}", f"{row['Count']} stocks")
            elif idx == 2:
                with col_sum3:
                    st.metric(f"{row['Cluster']}", f"{row['Count']} stocks")
        
        # Show stocks by cluster
        selected_cluster = st.selectbox("Select Cluster to View", df_clusters['Cluster_Label'].unique())
        cluster_stocks = df_clusters[df_clusters['Cluster_Label'] == selected_cluster]
        
        if not cluster_stocks.empty:
            st.dataframe(
                cluster_stocks[['Code', 'Sector', 'Smart_Flow_Miliar', 'Flow_Ratio', 'Volatility']]
                .sort_values('Smart_Flow_Miliar', ascending=False)
                .style.format({'Smart_Flow_Miliar': '{:.2f}', 'Flow_Ratio': '{:.1f}', 'Volatility': '{:.2f}'}),
                use_container_width=True, hide_index=True
            )
    else:
        st.info("Clustering data tidak tersedia.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 3: INSTITUTIONAL FOOTPRINT MAP ---
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🗺️ Institutional Footprint Analysis (60 Days)</div>', unsafe_allow_html=True)
    
    with st.spinner("Tracking institutional footprints..."):
        df_footprint = track_institutional_footprint(df, window_days=60)
    
    if not df_footprint.empty:
        # Footprint Map
        fig_footprint = px.scatter(df_footprint.head(30), 
                                  x='Total_Inst_Flow', 
                                  y='Flow_Percentage',
                                  size='Footprint_Score',
                                  color='Sector',
                                  hover_data=['Code', 'Price', 'Ownership_Change'],
                                  title="Institutional Footprint Map (Top 30)",
                                  size_max=40)
        
        # Add quadrant lines
        avg_flow = df_footprint['Total_Inst_Flow'].mean()
        avg_pct = df_footprint['Flow_Percentage'].mean()
        
        fig_footprint.add_hline(y=avg_pct, line_dash="dash", line_color="gray", 
                               annotation_text="Avg Flow %")
        fig_footprint.add_vline(x=avg_flow, line_dash="dash", line_color="gray",
                               annotation_text="Avg Flow Value")
        
        # Add quadrant annotations
        fig_footprint.add_annotation(x=avg_flow*1.5, y=avg_pct*1.5, 
                                    text="High Conviction", showarrow=False, font=dict(color="#05CD99"))
        fig_footprint.add_annotation(x=avg_flow*0.5, y=avg_pct*1.5, 
                                    text="Stealth Accumulation", showarrow=False, font=dict(color="#4318FF"))
        
        st.plotly_chart(update_plotly_layout(fig_footprint), use_container_width=True)
        
        # Top Footprint Changes
        st.markdown("#### 📈 Top Institutional Footprint Changes")
        st.dataframe(
            df_footprint.head(15)[['Code', 'Sector', 'Price', 'Total_Inst_Flow', 
                                  'Flow_Percentage', 'Footprint_Score']]
            .style.format({'Price': '{:,.0f}', 'Total_Inst_Flow': '{:,.0f}', 
                          'Flow_Percentage': '{:.2f}%', 'Footprint_Score': '{:.0f}'})
            .bar(subset=['Footprint_Score'], color='#4318FF'),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Tidak ada data footprint yang signifikan.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- SECTION 4: REAL-TIME DETECTION ---
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔍 Real-Time Pattern Detection</div>', unsafe_allow_html=True)
    
    col_detect1, col_detect2 = st.columns(2)
    
    with col_detect1:
        stock_to_check = st.selectbox("Check Specific Stock", df['Code'].unique()[:100])
    
    with col_detect2:
        detection_type = st.selectbox("Detection Type", ["Coordinated Accumulation", "Stealth Accumulation", "Full Analysis"])
    
    if st.button("🚀 Run Detection", use_container_width=True):
        with st.spinner("Analyzing patterns..."):
            if detection_type == "Coordinated Accumulation":
                is_coordinated, institutions = detect_coordinated_accumulation(df, stock_to_check)
                
                if is_coordinated:
                    st.success(f"✅ **COORDINATED ACCUMULATION DETECTED in {stock_to_check}**")
                    for inst in institutions:
                        st.markdown(f"""
                        **{inst['institution']}**
                        - Amount: {inst['amount_formatted']}
                        - Percentage of Shares: {inst['percentage']:.3f}%
                        """)
                else:
                    st.info("No coordinated accumulation detected.")
            
            elif detection_type == "Stealth Accumulation":
                is_stealth, details = detect_stealth_accumulation(df, stock_to_check)
                
                if is_stealth:
                    st.success(f"✅ **STEALTH ACCUMULATION DETECTED in {stock_to_check}**")
                    st.markdown(f"""
                    **Pattern Analysis:**
                    - Stealth Score: {details['stealth_score']:.0f}/100
                    - Positive Days: {details['positive_days']}/20 ({details['positive_days']/20*100:.0f}%)
                    - Total Accumulation: {format_id_short(details['total_stealth_accumulation'], True)}
                    - Average Daily: {format_id_short(details['avg_daily_flow'], True)}
                    - Max Single Day: {format_id_short(details['max_single_day_flow'], True)}
                    """)
                else:
                    st.info("No stealth accumulation detected.")
            
            else:  # Full Analysis
                # Conviction Score
                conviction, components = calculate_institutional_conviction(df, stock_to_check)
                
                # Coordinated
                is_coordinated, institutions = detect_coordinated_accumulation(df, stock_to_check)
                
                # Stealth
                is_stealth, stealth_details = detect_stealth_accumulation(df, stock_to_check)
                
                # Display Results
                st.markdown(f"## 📊 Complete Analysis for {stock_to_check}")
                
                col_full1, col_full2, col_full3 = st.columns(3)
                
                with col_full1:
                    st.metric("Conviction Score", f"{conviction:.0f}")
                    st.progress(conviction/100)
                
                with col_full2:
                    st.metric("Stealth Detection", "✅" if is_stealth else "❌")
                    if is_stealth:
                        st.caption(f"Score: {stealth_details['stealth_score']:.0f}/100")
                
                with col_full3:
                    st.metric("Coordinated", "✅" if is_coordinated else "❌")
                    if is_coordinated:
                        st.caption(f"{len(institutions)} institutions")
                
                # Score Components
                st.markdown("#### 📈 Score Components")
                comp_df = pd.DataFrame({
                    'Component': list(components.keys()),
                    'Score': list(components.values())
                })
                
                fig_comp = px.bar(comp_df, x='Component', y='Score', 
                                 color='Score', color_continuous_scale='Viridis')
                st.plotly_chart(update_plotly_layout(fig_comp), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 🎯 FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #2B3674; padding: 2rem;">
    <p>📊 <strong>KSEI Bandarmology PRO</strong> | Institutional Intelligence Platform</p>
    <p style="font-size: 0.9rem; color: #A3AED0;">
        Advanced detection of coordinated accumulation, stealth patterns & high-conviction plays
    </p>
</div>
""", unsafe_allow_html=True)
