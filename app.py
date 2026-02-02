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
from plotly.subplots import make_subplots

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

# ==============================================================================
# 🔥 FIX: UPDATE KONFIGURASI KOLOM BERDASARKAN HEADER BARU
# ==============================================================================

# --- KOLOM KEPEMILIKAN (STATIC) ---
OWNERSHIP_COLS = [
    'Local IS', 'Local CP', 'Local PF', 'Local IB', 'Local ID', 'Local MF', 'Local SC', 'Local FD', 'Local OT',
    'Foreign IS', 'Foreign CP', 'Foreign PF', 'Foreign IB', 'Foreign ID', 'Foreign MF', 'Foreign SC', 'Foreign FD', 'Foreign OT'
]

# --- KOLOM PERUBAHAN BERDASARKAN HEADER BARU ---
# Format baru: Local IS_Chg_Vol (Volume) dan Local IS_Chg_Val (Value/Rp)
OWNERSHIP_CHG_VOL_COLS = [f"{col}_Chg_Vol" for col in OWNERSHIP_COLS]
OWNERSHIP_CHG_VAL_COLS = [f"{col}_Chg_Val" for col in OWNERSHIP_COLS]

# Untuk backward compatibility, kita buat alias yang dipakai di code sebelumnya
OWNERSHIP_CHG_COLS = OWNERSHIP_CHG_VOL_COLS  # Alias untuk kolom volume
OWNERSHIP_CHG_RP_COLS = OWNERSHIP_CHG_VAL_COLS  # Alias untuk kolom value (Rp)

# --- SMART MONEY COLUMNS (UPDATE DENGAN FORMAT BARU) ---
SMART_MONEY_COLS = [
    'Foreign IS_Chg_Val', 'Foreign IB_Chg_Val', 'Foreign PF_Chg_Val', 
    'Local IS_Chg_Val', 'Local PF_Chg_Val', 'Local MF_Chg_Val', 'Local IB_Chg_Val',
    'Local CP_Chg_Val'  # Include Local CP as Smart Money
]

RETAIL_COLS = ['Local ID_Chg_Val']

# --- KOLOM LAIN YANG PERLU DIPERHATIKAN ---
# Pastikan kolom-kolom ini ada atau kita buat fallback
TOTAL_COLS = ['Total_Local', 'Total_Foreign']
TOP_PLAYER_COLS = ['Top_Buyer', 'Top_Buyer_Vol', 'Top_Buyer_Val', 'Top_Seller', 'Top_Seller_Vol', 'Top_Seller_Val']

# ==============================================================================
# 📦 3) FUNGSI MEMUAT DATA (DENGAN HANDLING KOLOM BARU)
# ==============================================================================
def get_gdrive_service():
    try:
        # Pengecekan standar untuk secrets
        if "gcp_service_account" not in st.secrets:
            return None, "❌ Key 'gcp_service_account' missing in secrets."
            
        creds_data = st.secrets["gcp_service_account"]
        if hasattr(creds_data, "to_dict"):
            creds_json = creds_data.to_dict()
        else:
            creds_json = dict(creds_data)
            
        # Fix Private Key Format (Just in case)
        if "private_key" in creds_json:
            pk = str(creds_json["private_key"])
            if "\\n" in pk:
                creds_json["private_key"] = pk.replace("\\n", "\n")

        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None
    except Exception as e:
        return None, f"❌ Auth Error: {e}"

@st.cache_data(ttl=3600, show_spinner="🔄 Memuat data...")
def load_data():
    service, error_msg = get_gdrive_service()
    if error_msg: 
        return pd.DataFrame(), error_msg, "error"

    try:
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1).execute()
        items = results.get('files', [])

        if not items: 
            return pd.DataFrame(), f"❌ File '{FILE_NAME}' not found.", "error"

        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: 
            status, done = downloader.next_chunk()
        fh.seek(0)

        df = pd.read_csv(fh, dtype=object)
        
        # 🔥 DEBUG: Tampilkan kolom yang ada
        print("Kolom yang tersedia:", df.columns.tolist())
        
        # Validasi kolom minimal
        required_cols = ['Date', 'Code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return pd.DataFrame(), f"❌ Missing required columns: {missing_cols}", "error"

        df.columns = df.columns.str.strip()
        
        # Konversi Date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Filter data dari 2023
        df = df[df['Date'].dt.year >= 2023].copy()
        
        if df.empty: 
            return pd.DataFrame(), "❌ No data found after 2023 filter.", "error"

        # Handle Sector column
        if 'Sector' in df.columns:
            df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
            df['Sector'] = 'Others'

        # 🔥 DAFTAR KOLOM YANG AKAN DIKONVERSI KE NUMERIC
        cols_to_numeric = [
            'Price', 'Price_Chg %', 'Free Float', 'Sec. Num', 'Avg_Price', 'Total_Shares',
            'Local IS', 'Local CP', 'Local PF', 'Local IB', 'Local ID', 'Local MF', 
            'Local SC', 'Local FD', 'Local OT', 'Total_Local',
            'Foreign IS', 'Foreign CP', 'Foreign PF', 'Foreign IB', 'Foreign ID', 
            'Foreign MF', 'Foreign SC', 'Foreign FD', 'Foreign OT', 'Total_Foreign'
        ]
        
        # Tambahkan kolom perubahan jika ada
        all_change_cols = OWNERSHIP_CHG_VOL_COLS + OWNERSHIP_CHG_VAL_COLS + TOP_PLAYER_COLS
        cols_to_numeric.extend([col for col in all_change_cols if col in df.columns])
        
        # Konversi ke numeric
        for col in cols_to_numeric:
            if col in df.columns:
                # Handle berbagai format
                cleaned = df[col].astype(str).str.strip()
                # Hapus koma, karakter non-numeric, dll
                cleaned = cleaned.str.replace(',', '', regex=False)
                cleaned = cleaned.str.replace('[^\d.-]', '', regex=True)
                df[col] = pd.to_numeric(cleaned, errors='coerce').fillna(0)
        
        # Drop rows dengan Date atau Code kosong
        df = df.dropna(subset=['Date', 'Code'])
        
        # Hapus duplikat
        if df.duplicated(subset=['Date', 'Code']).any():
            df = df.drop_duplicates(subset=['Date', 'Code'], keep='last')
        
        # 🔥 CALCULATE DERIVED COLUMNS (untuk backward compatibility)
        # Hitung Total_Local_chg_Rp dan Total_Foreign_chg_Rp jika belum ada
        if 'Total_Local_chg_Rp' not in df.columns:
            local_chg_cols = [c for c in OWNERSHIP_CHG_VAL_COLS if 'Local' in c and c in df.columns]
            if local_chg_cols:
                df['Total_Local_chg_Rp'] = df[local_chg_cols].sum(axis=1)
            else:
                df['Total_Local_chg_Rp'] = 0
        
        if 'Total_Foreign_chg_Rp' not in df.columns:
            foreign_chg_cols = [c for c in OWNERSHIP_CHG_VAL_COLS if 'Foreign' in c and c in df.columns]
            if foreign_chg_cols:
                df['Total_Foreign_chg_Rp'] = df[foreign_chg_cols].sum(axis=1)
            else:
                df['Total_Foreign_chg_Rp'] = 0
        
        # Total change Rp
        df['Total_chg_Rp'] = df['Total_Local_chg_Rp'] + df['Total_Foreign_chg_Rp']
        
        # 🔥 FIX: Untuk backward compatibility dengan kode yang masih pakai nama lama
        # Buat alias untuk kolom Top_Buyer_Value_Rp dan Top_Seller_Value_Rp
        if 'Top_Buyer_Val' in df.columns and 'Top_Buyer_Value_Rp' not in df.columns:
            df['Top_Buyer_Value_Rp'] = df['Top_Buyer_Val']
        
        if 'Top_Seller_Val' in df.columns and 'Top_Seller_Value_Rp' not in df.columns:
            df['Top_Seller_Value_Rp'] = df['Top_Seller_Val']

        return df, "✅ Data Loaded Successfully", "success"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return pd.DataFrame(), f"❌ Error loading data: {str(e)}", "error"

# ==============================================================================
# 🛠️ 4) HELPER FUNCTIONS (REVISI)
# ==============================================================================

def format_id_short(value, is_currency=False):
    if pd.isna(value) or value == 0: 
        return "0"
    
    val_abs = abs(value)
    
    if val_abs >= 1e12:
        suffix, divisor = " T", 1e12
    elif val_abs >= 1e9:
        suffix, divisor = " M", 1e9
    elif val_abs >= 1e6:
        suffix, divisor = " Jt", 1e6
    else:
        return f"{value:,.0f}"
    
    formatted = f"{value/divisor:.1f}"
    if formatted.endswith(".0"):
        formatted = formatted[:-2]
    
    prefix = "Rp " if is_currency else ""
    return f"{prefix}{formatted}{suffix}"

def update_plotly_layout(fig):
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2B3674', family='DM Sans'),
        xaxis=dict(gridcolor='#E0E5F2', showgrid=True, tickfont=dict(color='#A3AED0')), 
        yaxis=dict(gridcolor='#E0E5F2', showgrid=True, tickfont=dict(color='#A3AED0')),
        margin=dict(t=40, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# 🔥 PERBAIKAN: Fungsi get_available_columns yang bisa handle Series dan DataFrame
def get_available_columns(data, expected_cols):
    """Helper untuk mendapatkan kolom yang tersedia dari daftar expected"""
    if isinstance(data, pd.DataFrame):
        return [col for col in expected_cols if col in data.columns]
    elif isinstance(data, pd.Series):
        return [col for col in expected_cols if col in data.index]
    else:
        # Fallback: coba akses sebagai object
        return [col for col in expected_cols if hasattr(data, col)]

# ==============================================================================
# 📊 6) EXISTING HELPER FUNCTIONS (REVISI LAGI)
# ==============================================================================

@st.cache_data
def calculate_macro_flow(df_filtered):
    """Hitung flow makro - PERBAIKAN UTAMA DI SINI"""
    # 1. Validasi input
    if df_filtered.empty:
        # Return empty dataframes dengan struktur yang benar
        net_flow = pd.DataFrame(columns=['Kategori', 'Total Net Flow (Rp)'])
        cum_flow = pd.DataFrame(columns=['Date', 'Kategori', 'Cumulative Flow (Rp)'])
        return net_flow, cum_flow
    
    # 2. Hitung net flow per kategori (gunakan kolom value/_Chg_Val)
    available_val_cols = []
    
    # Coba format baru dulu (_Chg_Val)
    for col in OWNERSHIP_CHG_VAL_COLS:
        if col in df_filtered.columns:
            available_val_cols.append(col)
    
    # Jika tidak ada kolom baru, coba kolom lama untuk backward compatibility
    if not available_val_cols:
        # Coba format lama (_chg_Rp)
        old_format_cols = [f"{col}_chg_Rp" for col in OWNERSHIP_COLS]
        for col in old_format_cols:
            if col in df_filtered.columns:
                available_val_cols.append(col)
    
    if not available_val_cols:
        # Jika masih tidak ada, coba kolom apa saja yang mengandung '_Chg' atau 'chg'
        possible_cols = [col for col in df_filtered.columns if '_Chg' in col or 'chg' in col.lower()]
        available_val_cols = possible_cols
    
    if available_val_cols:
        try:
            # Hitung sum untuk setiap kolom
            net_flow_series = df_filtered[available_val_cols].sum()
            
            # Convert ke DataFrame
            net_flow = pd.DataFrame({
                'Kategori': net_flow_series.index,
                'Total Net Flow (Rp)': net_flow_series.values
            })
            
            # Clean up kategori names
            net_flow['Kategori'] = net_flow['Kategori'].str.replace('_Chg_Val', '')
            net_flow['Kategori'] = net_flow['Kategori'].str.replace('_chg_Rp', '')
            net_flow['Kategori'] = net_flow['Kategori'].str.replace('_chg', '')
            
            net_flow = net_flow.sort_values('Total Net Flow (Rp)', ascending=False).reset_index(drop=True)
        except Exception as e:
            print(f"Error calculating net flow: {e}")
            net_flow = pd.DataFrame(columns=['Kategori', 'Total Net Flow (Rp)'])
    else:
        net_flow = pd.DataFrame(columns=['Kategori', 'Total Net Flow (Rp)'])
    
    # 3. Hitung cumulative flow (local vs foreign)
    try:
        # Cek kolom derived
        local_col = 'Total_Local_chg_Rp' if 'Total_Local_chg_Rp' in df_filtered.columns else None
        foreign_col = 'Total_Foreign_chg_Rp' if 'Total_Foreign_chg_Rp' in df_filtered.columns else None
        
        if local_col and foreign_col:
            # Hitung cumulative sum
            cum_data = df_filtered.groupby('Date')[[local_col, foreign_col]].sum().cumsum().reset_index()
            
            # Melt untuk plotting
            cum_flow = pd.melt(
                cum_data, 
                id_vars=['Date'],
                value_vars=[local_col, foreign_col],
                var_name='Kategori',
                value_name='Cumulative Flow (Rp)'
            )
            
            # Clean up kategori names
            cum_flow['Kategori'] = cum_flow['Kategori'].str.replace('_chg_Rp', ' (Net Rp)')
            cum_flow['Kategori'] = cum_flow['Kategori'].str.replace('Total_', '')
        else:
            # Coba hitung manual dari kolom individual
            local_val_cols = [col for col in available_val_cols if 'Local' in col]
            foreign_val_cols = [col for col in available_val_cols if 'Foreign' in col]
            
            if local_val_cols and foreign_val_cols:
                # Calculate daily totals
                df_filtered['Local_Daily'] = df_filtered[local_val_cols].sum(axis=1)
                df_filtered['Foreign_Daily'] = df_filtered[foreign_val_cols].sum(axis=1)
                
                # Group by date and calculate cumulative
                daily_totals = df_filtered.groupby('Date')[['Local_Daily', 'Foreign_Daily']].sum()
                cum_totals = daily_totals.cumsum().reset_index()
                
                cum_flow = pd.melt(
                    cum_totals,
                    id_vars=['Date'],
                    value_vars=['Local_Daily', 'Foreign_Daily'],
                    var_name='Kategori',
                    value_name='Cumulative Flow (Rp)'
                )
                
                cum_flow['Kategori'] = cum_flow['Kategori'].str.replace('_Daily', ' (Net Rp)')
            else:
                cum_flow = pd.DataFrame(columns=['Date', 'Kategori', 'Cumulative Flow (Rp)'])
                
    except Exception as e:
        print(f"Error calculating cumulative flow: {e}")
        cum_flow = pd.DataFrame(columns=['Date', 'Kategori', 'Cumulative Flow (Rp)'])
    
    return net_flow, cum_flow

@st.cache_data
def calculate_sector_rotation(df_filtered, selected_category):
    """Hitung rotasi sektor untuk kategori tertentu"""
    # Formatkan nama kolom sesuai format baru
    col = f"{selected_category}_Chg_Val"
    
    if col not in df_filtered.columns:
        return pd.DataFrame(), f"Kolom '{col}' tidak ditemukan."
    
    sector_flow = df_filtered.groupby('Sector')[col].sum().reset_index()
    sector_flow.columns = ['Sector', 'Net Flow (Rp)']
    return sector_flow.sort_values('Net Flow (Rp)', ascending=False), None

@st.cache_data
def get_stock_ownership_state(df, stock_code):
    """Ambil data kepemilikan terbaru untuk satu saham"""
    df_stock = df[df['Code'] == stock_code]
    if df_stock.empty: 
        return pd.DataFrame(), pd.Series(dtype='object')
    
    latest = df_stock.sort_values('Date').iloc[-1]
    
    # Ambil hanya kolom ownership yang tersedia
    available_ownership_cols = get_available_columns(latest, OWNERSHIP_COLS)
    df_state = latest[available_ownership_cols].reset_index()
    df_state.columns = ['Kategori', 'Jumlah Saham']
    
    return df_state, latest

@st.cache_data
def calculate_monthly_change_table(df_stock):
    """Hitung perubahan bulanan (Volume)"""
    # Gunakan kolom volume (Chg_Vol)
    available_vol_cols = get_available_columns(df_stock, OWNERSHIP_CHG_VOL_COLS)
    
    if not available_vol_cols:
        return pd.DataFrame(columns=['Bulan'] + OWNERSHIP_CHG_VOL_COLS)
    
    df_res = df_stock.sort_values('Date', ascending=False)[['Date'] + available_vol_cols].copy()
    df_res.rename(columns={'Date': 'Bulan'}, inplace=True)
    df_res['Bulan'] = df_res['Bulan'].dt.strftime('%b %Y')
    return df_res

@st.cache_data
def calculate_smart_money_signals(df_year, window_periods=2, min_acc_threshold=20e9):
    """Hitung sinyal smart money"""
    if df_year.empty: 
        return pd.DataFrame()
    
    results = []
    
    # Filter only stocks with data
    valid_codes = df_year['Code'].unique()
    
    for code in valid_codes:
        df_w = df_year[df_year['Code'] == code].sort_values('Date').tail(window_periods)
        if df_w.empty: 
            continue
        
        last_p = df_w.iloc[-1]['Price']
        start_p = df_w.iloc[0]['Price']
        pct = ((last_p - start_p)/start_p)*100 if start_p > 0 else 0
        
        # Gunakan kolom value untuk smart money dan retail
        available_sm_cols = get_available_columns(df_w, SMART_MONEY_COLS)
        available_ret_cols = get_available_columns(df_w, RETAIL_COLS)
        
        sm_sum = df_w[available_sm_cols].sum().sum() if available_sm_cols else 0
        ret_sum = df_w[available_ret_cols].sum().sum() if available_ret_cols else 0
        
        status = "Netral"
        if sm_sum > min_acc_threshold and ret_sum < 0: 
            status = "🔥 Big Accumulation"
        elif sm_sum > (min_acc_threshold/2) and pct <= 5: 
            status = "💎 Divergence"
        elif sm_sum < -min_acc_threshold and ret_sum > 0: 
            status = "⚠️ Distribution"
        
        if status != "Netral":
            results.append({
                'Code': code, 
                'Sector': df_w.iloc[-1].get('Sector','N/A'),
                'Price': last_p, 
                'Price Chg %': pct,
                'Smart Money (Rp)': sm_sum, 
                'Retail (Rp)': ret_sum, 
                'Signal': status
            })
    
    df_res = pd.DataFrame(results)
    if not df_res.empty: 
        df_res = df_res.sort_values('Smart Money (Rp)', ascending=False)
    return df_res

@st.cache_data
def get_significant_movements(df_month, threshold_rp=50e9, threshold_pct=1):
    """Dapatkan saham dengan pergerakan signifikan"""
    if df_month.empty: 
        return pd.DataFrame()
    
    results = []
    # Gunakan kolom value (Rp)
    available_rp_cols = get_available_columns(df_month, OWNERSHIP_CHG_VAL_COLS)

    for code in df_month['Code'].unique():
        row = df_month[df_month['Code'] == code].iloc[0]
        abs_flow = sum(abs(row[c]) for c in available_rp_cols)
        net_flow = row.get('Total_chg_Rp', 0)
        shares = row.get('Sec. Num', 1)
        pct = (abs_flow / shares * 100) if shares > 0 else 0
        
        if abs_flow >= threshold_rp or pct >= threshold_pct:
            direction = "NET BUY" if net_flow > 0 else "NET SELL" if net_flow < 0 else "NEUTRAL"
            
            # Cari top buyer dan seller
            vals = {c.replace('_Chg_Val', ''): row[c] for c in available_rp_cols if c in row}
            top_b_cat = max(vals, key=vals.get) if vals and max(vals.values()) > 0 else "-"
            top_s_cat = min(vals, key=vals.get) if vals and min(vals.values()) < 0 else "-"
            
            buyer_str = f"{top_b_cat}" if top_b_cat != "-" and vals[top_b_cat] > 0 else "-"
            seller_str = f"{top_s_cat}" if top_s_cat != "-" and vals[top_s_cat] < 0 else "-"

            results.append({
                'Code': code, 
                'Sector': row.get('Sector','N/A'),
                'Price': row.get('Price',0), 
                'Total Flow (Rp)': abs_flow,
                'Net Flow (Rp)': net_flow, 
                'Flow %': pct, 
                'Direction': direction,
                'Top_Buyer': buyer_str, 
                'Top_Seller': seller_str
            })
    
    df_res = pd.DataFrame(results)
    if not df_res.empty: 
        df_res = df_res.sort_values('Total Flow (Rp)', ascending=False)
    return df_res

def create_sankey_chart(df, stock_code, selected_date, mode='Volume'):
    """Buat diagram Sankey - PERBAIKAN: handle kolom baru"""
    row = df[(df['Code'] == stock_code) & (df['Date'] == selected_date)]
    if row.empty: 
        return None
    
    row = row.iloc[0]
    
    # Tentukan kolom berdasarkan mode
    if mode == 'Volume':
        cols = get_available_columns(row, OWNERSHIP_CHG_VOL_COLS)
        is_rp = False
    else:  # Value mode
        cols = get_available_columns(row, OWNERSHIP_CHG_VAL_COLS)
        is_rp = True
    
    if not cols:
        return None
    
    sellers, buyers, total_vol = [], [], 0
    
    for col in cols:
        val = row[col]
        # Format nama kategori (hilangkan suffix)
        if '_Chg_Vol' in col:
            cat = col.replace('_Chg_Vol', '')
        elif '_Chg_Val' in col:
            cat = col.replace('_Chg_Val', '')
        else:
            cat = col
        
        if val != 0:
            lbl = f"{cat}\n({format_id_short(abs(val), is_rp)})"
            if val < 0: 
                sellers.append({'l': lbl, 'v': abs(val)})
                total_vol += abs(val)
            else: 
                buyers.append({'l': lbl, 'v': val})
                
    if not sellers and not buyers: 
        return None
    
    labels = [f"MARKET\n({format_id_short(total_vol, is_rp)})"] + [s['l'] for s in sellers] + [b['l'] for b in buyers]
    colors = ["#E0E5F2"] + ["#EE5D50"]*len(sellers) + ["#05CD99"]*len(buyers) 
    
    source = list(range(1, len(sellers)+1)) + [0]*len(buyers)
    target = [0]*len(sellers) + list(range(len(sellers)+1, len(labels)))
    values = [s['v'] for s in sellers] + [b['v'] for b in buyers]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=20, thickness=15, line=dict(color="white", width=0.5), label=labels, color=colors),
        link=dict(source=source, target=target, value=values, 
                color=['rgba(238, 93, 80, 0.4)']*len(sellers) + ['rgba(5, 205, 153, 0.4)']*len(buyers))
    )])
    
    mode_text = "Value (Rp)" if mode == 'Value' else "Volume (Lot)"
    fig.update_layout(title_text=f"Arus Dana {stock_code} ({selected_date.strftime('%b %Y')}) - {mode_text}", 
                     font_size=14, height=500)
    return update_plotly_layout(fig)


# ==============================================================================
# 🧠 5) ADVANCED INTELLIGENCE LOGIC (THE MISSING BRAINS)
# ==============================================================================

@st.cache_data
def calculate_institutional_conviction(df, stock_code, window=3):
    """
    Menghitung skor keyakinan institusi (0-100).
    Logic: Konsistensi Net Buy + Besaran Flow vs Market Cap + Price Action.
    """
    df_stock = df[df['Code'] == stock_code].sort_values('Date').tail(window)
    if df_stock.empty: return 0, {}
    
    # 1. Consistency Score (Berapa kali Net Buy dalam window?)
    # Gunakan kolom Smart Money
    sm_cols = get_available_columns(df_stock, SMART_MONEY_COLS)
    if not sm_cols: return 0, {}
    
    df_stock['SM_Net'] = df_stock[sm_cols].sum(axis=1)
    consistency = (df_stock['SM_Net'] > 0).sum() / len(df_stock) * 40 # Max 40 poin
    
    # 2. Flow Magnitude Score (Total Flow vs Rata2 Transaksi - simulasi)
    # Karena tidak ada data volume transaksi harian, kita pakai absolut flow
    total_flow = df_stock['SM_Net'].sum()
    magnitude = min(40, (total_flow / 10_000_000_000) * 10) # 10 Miliar = 10 poin, Max 40
    if total_flow < 0: magnitude = 0 
    
    # 3. Price Divergence Score (Bonus jika Flow Positif tapi Harga Belum Naik Banyak)
    price_chg = df_stock.iloc[-1]['Price'] - df_stock.iloc[0]['Price']
    price_chg_pct = (price_chg / df_stock.iloc[0]['Price']) * 100 if df_stock.iloc[0]['Price'] > 0 else 0
    
    divergence = 0
    if total_flow > 0:
        if -5 <= price_chg_pct <= 5: # Akumulasi di area sideways
            divergence = 20
        elif price_chg_pct > 5: # Markup phase
            divergence = 10
            
    total_score = consistency + magnitude + divergence
    
    details = {
        'Consistency': consistency,
        'Magnitude': magnitude,
        'Divergence': divergence,
        'Total_Flow': total_flow
    }
    
    return min(100, max(0, total_score)), details

@st.cache_data
def detect_stealth_accumulation(df, stock_code, window=2):
    """
    Mendeteksi akumulasi senyap: Flow Besar, Harga Flat/Turun Tipis.
    """
    df_stock = df[df['Code'] == stock_code].sort_values('Date').tail(window)
    if df_stock.empty: return False, {}
    
    sm_cols = get_available_columns(df_stock, SMART_MONEY_COLS)
    if not sm_cols: return False, {}
    
    net_sm_flow = df_stock[sm_cols].sum().sum()
    
    start_p = df_stock.iloc[0]['Price']
    end_p = df_stock.iloc[-1]['Price']
    pct_chg = ((end_p - start_p) / start_p * 100) if start_p > 0 else 0
    
    # LOGIC STEALTH:
    # 1. Flow Institusi Positif Signifikan (> 5 Miliar misalnya)
    # 2. Kenaikan harga KECIL (< 3%) atau malah minus dikit
    is_stealth = (net_sm_flow > 5_000_000_000) and (pct_chg < 3) and (pct_chg > -5)
    
    return is_stealth, {'Net_Flow': net_sm_flow, 'Pct_Chg': pct_chg}

@st.cache_data
def scan_high_conviction_stocks(df, min_score=70, min_flow=5e9):
    """
    Scanning seluruh saham untuk mencari High Conviction Score.
    """
    # Ambil data bulan terakhir saja untuk efisiensi
    latest_date = df['Date'].max()
    # Kita butuh window beberapa bulan ke belakang
    start_date = latest_date - pd.DateOffset(months=2) 
    df_recent = df[df['Date'] >= start_date]
    
    results = []
    unique_codes = df_recent['Code'].unique()
    
    for code in unique_codes:
        score, details = calculate_institutional_conviction(df_recent, code)
        is_stealth, _ = detect_stealth_accumulation(df_recent, code)
        
        # Check thresholds
        if score >= min_score and details.get('Total_Flow', 0) >= min_flow:
            # Ambil data terakhir untuk display
            last_row = df_recent[df_recent['Code'] == code].iloc[-1]
            
            results.append({
                'Code': code,
                'Sector': last_row.get('Sector', 'Others'),
                'Price': last_row.get('Price', 0),
                'Price_Chg_%': last_row.get('Price_Chg %', 0),
                'Conviction_Score': score,
                'Institutional_Flow': details['Total_Flow'],
                'Is_Stealth': is_stealth,
                'Is_Coordinated': False # Placeholder for future logic
            })
            
    return pd.DataFrame(results).sort_values('Conviction_Score', ascending=False)

@st.cache_data
def cluster_smart_money_patterns(df, n_clusters=4, sample_size=300):
    """
    Menggunakan K-Means untuk mengelompokkan saham berdasarkan perilaku Smart Money.
    Features: Smart Money Flow, Retail Flow, Price Volatility.
    """
    # Ambil data agregat per saham (3 bulan terakhir)
    latest_date = df['Date'].max()
    start_date = latest_date - pd.DateOffset(months=3)
    df_recent = df[df['Date'] >= start_date].copy()
    
    # Prepare Features per Stock
    stock_features = []
    
    sm_cols = get_available_columns(df_recent, SMART_MONEY_COLS)
    ret_cols = get_available_columns(df_recent, RETAIL_COLS)
    
    if not sm_cols: return pd.DataFrame()

    grouped = df_recent.groupby('Code')
    
    for code, data in grouped:
        if len(data) < 2: continue
        
        sm_flow = data[sm_cols].sum().sum()
        ret_flow = data[ret_cols].sum().sum() if ret_cols else 0
        market_cap_proxy = data['Price'].mean() * data['Total_Shares'].mean() if 'Total_Shares' in data.columns else 1e12
        
        # Volatility (Std Dev of Price / Mean Price)
        volatility = data['Price'].std() / data['Price'].mean() if data['Price'].mean() > 0 else 0
        
        # Ratio Flow terhadap Market Cap (agar saham kecil vs besar apple-to-apple)
        # Karena market cap proxy kasar, kita pakai log flow saja atau raw flow untuk sample sejenis
        
        stock_features.append({
            'Code': code,
            'Sector': data.iloc[-1].get('Sector', 'Others'),
            'Smart_Flow_Raw': sm_flow,
            'Retail_Flow_Raw': ret_flow,
            'Volatility': volatility,
            'Flow_Ratio': sm_flow / market_cap_proxy if market_cap_proxy > 0 else 0
        })
    
    df_feat = pd.DataFrame(stock_features)
    if df_feat.empty: return pd.DataFrame()
    
    # Filter sample size (top liquid / top flow)
    df_feat['Abs_Flow'] = df_feat['Smart_Flow_Raw'].abs()
    df_feat = df_feat.nlargest(sample_size, 'Abs_Flow')
    
    # Normalize Data for K-Means
    features = ['Smart_Flow_Raw', 'Volatility']
    scaler = StandardScaler()
    X = scaler.fit_transform(df_feat[features].fillna(0))
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_feat['Cluster'] = kmeans.fit_predict(X)
    
    # Labeling Clusters (Automated Naming)
    # Kita cari rata-rata flow per cluster untuk menamai
    cluster_stats = df_feat.groupby('Cluster')['Smart_Flow_Raw'].mean().sort_values(ascending=False)
    
    # Assign names based on rank
    labels = {}
    sorted_clusters = cluster_stats.index.tolist()
    
    if n_clusters >= 3:
        labels[sorted_clusters[0]] = "🚀 Strong Accumulation"
        labels[sorted_clusters[-1]] = "⚠️ Big Distribution"
        
        # Tengah-tengah
        for c in sorted_clusters[1:-1]:
            mean_flow = cluster_stats[c]
            mean_vol = df_feat[df_feat['Cluster'] == c]['Volatility'].mean()
            if mean_flow > 0:
                labels[c] = "🕵️ Stealth Accumulation" if mean_vol < df_feat['Volatility'].mean() else "⚔️ Big Fight"
            else:
                labels[c] = "📊 Sideways/Retail"
    
    df_feat['Cluster_Label'] = df_feat['Cluster'].map(labels).fillna("Unknown")
    df_feat['Smart_Flow_Miliar'] = df_feat['Smart_Flow_Raw'] / 1e9
    
    return df_feat

@st.cache_data
def track_institutional_footprint(df, window_days=90):
    """
    Tracking pergerakan kumulatif institusi dalam window tertentu.
    """
    latest_date = df['Date'].max()
    start_date = latest_date - pd.DateOffset(days=window_days)
    df_window = df[df['Date'] >= start_date]
    
    sm_cols = get_available_columns(df_window, SMART_MONEY_COLS)
    if not sm_cols: return pd.DataFrame()
    
    results = []
    for code, data in df_window.groupby('Code'):
        total_inst_flow = data[sm_cols].sum().sum()
        
        # Score sederhana: Flow dibagi harga (proxy volume power)
        last_price = data.iloc[-1]['Price']
        if last_price == 0: continue
            
        # Berapa % flow ini terhadap estimasi nilai (kasar)
        # Footprint score 0-100
        score = min(100, (total_inst_flow / (last_price * 1_000_000)) * 10) # Arbitrary scaling
        if total_inst_flow < 0: score = 0
        
        results.append({
            'Code': code,
            'Sector': data.iloc[-1].get('Sector', 'Others'),
            'Price': last_price,
            'Total_Inst_Flow': total_inst_flow,
            'Flow_Percentage': (total_inst_flow / 1e11) * 100, # Dummy pct for visual
            'Footprint_Score': score
        })
        
    return pd.DataFrame(results).sort_values('Total_Inst_Flow', ascending=False)

# ==============================================================================
# 💎 7) LAYOUT UTAMA & SIDEBAR
# ==============================================================================

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=50)
    st.markdown("<h3 style='color:#2B3674; margin-top:0;'>Bandarmology PRO</h3>", unsafe_allow_html=True)
    st.divider()

    if st.button("🔄 Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Load data
    df, status_msg, status_level = load_data()
    
    # Debug info
    with st.expander("🔧 Debug Info"):
        if not df.empty:
            st.write(f"Rows: {len(df)}")
            st.write(f"Columns: {len(df.columns)}")
            st.write(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            # Tampilkan beberapa kolom yang penting
            important_cols = ['Date', 'Code', 'Sector', 'Price', 'Total_chg_Rp']
            st.write("Sample data:", df[important_cols].head(3) if all(col in df.columns for col in important_cols) else "Missing columns")
    
    if status_level == "error":
        st.error(status_msg)
        st.stop()
    else:
        st.success(status_msg)
    
    # Month Filter
    df['Date'] = pd.to_datetime(df['Date'])
    available_months = sorted(df['Date'].dt.strftime('%Y-%m').unique(), reverse=True)
    
    if available_months:
        selected_month_str = st.selectbox("📅 Pilih Bulan", available_months)
        selected_month = pd.to_datetime(selected_month_str + '-01')
        df_filtered_month = df[df['Date'].dt.strftime('%Y-%m') == selected_month_str]
    else:
        st.warning("Tidak ada data yang tersedia")
        selected_month_str = None
        df_filtered_month = pd.DataFrame()
    
    st.divider()
    
    with st.expander("⚙️ Advanced Filter"):
        threshold_rp = st.number_input("Min Flow (Rp)", value=50_000_000_000, step=10_000_000_000, format="%d")
        min_rotation = st.number_input("Min Rotation (Rp)", value=20_000_000_000, step=5_000_000_000, format="%d")
        conviction_threshold = st.slider("🎯 Conviction Threshold", min_value=50, max_value=100, value=75, step=5)

# --- MAIN PAGE HEADER ---
st.markdown(f"""
    <div class="header-banner">
        <div class="header-title">KSEI Bandarmology PRO - Data Bulanan</div>
        <div class="header-subtitle">Analisis kepemilikan institusional data bulanan KSEI | Format: _Chg_Vol / _Chg_Val</div>
    </div>
""", unsafe_allow_html=True)

# Tampilkan info kolom yang tersedia
if not df.empty:
    with st.expander("📊 Kolom Tersedia", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Ownership Columns:**")
            available_ownership = get_available_columns(df, OWNERSHIP_COLS)
            st.write(f"{len(available_ownership)}/{len(OWNERSHIP_COLS)} available")
        
        with col2:
            st.write("**Change Volume Columns:**")
            available_vol = get_available_columns(df, OWNERSHIP_CHG_VOL_COLS)
            st.write(f"{len(available_vol)}/{len(OWNERSHIP_CHG_VOL_COLS)} available")
        
        with col3:
            st.write("**Change Value Columns:**")
            available_val = get_available_columns(df, OWNERSHIP_CHG_VAL_COLS)
            st.write(f"{len(available_val)}/{len(OWNERSHIP_CHG_VAL_COLS)} available")

# ==============================================================================
# 📑 TABS VISUALISASI
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌍 Market Overview", "🏭 Sector Rotation", "📈 Deep Dive", "🔍 Big Screener", 
    "🤖 Smart Signals", "🔥 Top Movers", "🎯 Institutional Intel"
])

# --- TAB 1: MAKRO ---
with tab1:
    if not df.empty:
        col_m1, col_m2 = st.columns([2, 1])
        df_net, df_cum = calculate_macro_flow(df)
        
        with col_m1:
            st.markdown('<div class="css-card"><div class="card-title">📊 Cumulative Flow YTD (Rp)</div>', unsafe_allow_html=True)
            if not df_cum.empty:
                fig_macro = px.line(df_cum, x='Date', y='Cumulative Flow (Rp)', color='Kategori', 
                                    color_discrete_map={'Total_Local (Net Rp)': '#05CD99', 'Total_Foreign (Net Rp)': '#EE5D50'})
                st.plotly_chart(update_plotly_layout(fig_macro), use_container_width=True)
            else:
                st.info("Data cumulative flow tidak tersedia")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_m2:
            st.markdown('<div class="css-card"><div class="card-title">🏆 Top Net Buy (YTD)</div>', unsafe_allow_html=True)
            if not df_net.empty and len(df_net) > 0:
                df_net_buy = df_net[df_net['Total Net Flow (Rp)'] > 0].head(7)
                if not df_net_buy.empty:
                    fig_buy = px.bar(df_net_buy, x='Total Net Flow (Rp)', y='Kategori', orientation='h', color_discrete_sequence=['#05CD99'])
                    fig_buy.update_layout(height=300, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(update_plotly_layout(fig_buy), use_container_width=True)
                else:
                    st.info("Tidak ada net buy signifikan")
            else:
                st.info("Data net flow tidak tersedia")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="css-card"><div class="card-title">🔻 Top Net Sell (YTD)</div>', unsafe_allow_html=True)
            if not df_net.empty and len(df_net) > 0:
                df_net_sell = df_net[df_net['Total Net Flow (Rp)'] < 0].tail(7)
                if not df_net_sell.empty:
                    fig_sell = px.bar(df_net_sell, x='Total Net Flow (Rp)', y='Kategori', orientation='h', color_discrete_sequence=['#EE5D50'])
                    fig_sell.update_layout(height=300, yaxis={'categoryorder':'total descending'})
                    st.plotly_chart(update_plotly_layout(fig_sell), use_container_width=True)
                else:
                    st.info("Tidak ada net sell signifikan")
            else:
                st.info("Data net flow tidak tersedia")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Data tidak tersedia")

# --- TAB 2: SEKTOR ---
with tab2:
    if not df_filtered_month.empty and selected_month_str:
        col_sel, col_chart = st.columns([1, 4])
        
        with col_sel:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Filter</div>', unsafe_allow_html=True)
            
            # Dapatkan kategori yang tersedia
            available_cats = []
            for cat in OWNERSHIP_COLS:
                if f"{cat}_Chg_Val" in df.columns:
                    available_cats.append(cat)
            
            if available_cats:
                sel_cat = st.selectbox("Pilih Investor:", available_cats, 
                                      index=available_cats.index('Foreign IB') if 'Foreign IB' in available_cats else 0)
                df_sec, _ = calculate_sector_rotation(df_filtered_month, sel_cat)
                if not df_sec.empty:
                    total_flow = df_sec['Net Flow (Rp)'].sum()
                    st.metric("Total Flow", f"Rp {format_id_short(total_flow)}")
                else:
                    st.info("Data sektor tidak tersedia")
            else:
                st.warning("Tidak ada data investor tersedia")
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart:
            if 'df_sec' in locals() and not df_sec.empty:
                st.markdown(f'<div class="css-card"><div class="card-title">Rotasi Sektor {sel_cat} ({selected_month_str})</div>', unsafe_allow_html=True)
                fig_sec = px.bar(df_sec, x='Net Flow (Rp)', y='Sector', orientation='h', 
                                color='Net Flow (Rp)', color_continuous_scale='RdYlGn')
                fig_sec.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(update_plotly_layout(fig_sec), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Data tidak tersedia untuk visualisasi")
    else:
        st.warning(f"Tidak ada data untuk bulan {selected_month_str}")

# --- TAB 3: INDIVIDUAL ---
with tab3:
    if not df.empty:
        stocks_avail = sorted(df['Code'].unique())
        
        if stocks_avail:
            sel_stock = st.selectbox("🔎 Cari Kode Saham:", stocks_avail, 
                                   index=stocks_avail.index('BBRI') if 'BBRI' in stocks_avail else 0,
                                   key="tab3_stock")
            
            if sel_stock:
                df_stock_all = df[df['Code'] == sel_stock].sort_values('Date')
                
                if selected_month_str:
                    df_stock_month = df_stock_all[df_stock_all['Date'].dt.strftime('%Y-%m') == selected_month_str]
                else:
                    df_stock_month = pd.DataFrame()
                
                if df_stock_month.empty and not df_stock_all.empty:
                    st.warning(f"Data {sel_stock} bulan {selected_month_str} kosong. Menggunakan data terakhir tersedia.")
                    df_state, last_row = get_stock_ownership_state(df_stock_all, sel_stock)
                    date_for_display = df_stock_all['Date'].max().strftime('%b %Y')
                elif not df_stock_month.empty:
                    df_state, last_row = get_stock_ownership_state(df_stock_month, sel_stock)
                    date_for_display = selected_month_str
                else:
                    st.error(f"Tidak ada data untuk saham {sel_stock}")
                    st.stop()
                
                # Display metrics dengan try-except untuk handling error
                k1, k2, k3, k4 = st.columns(4)
                
                # Metric 1: Harga
                try:
                    price = float(last_row.get('Price', 0))
                    k1.markdown(f"""<div class="css-card" style="text-align:center;"><div style="font-size:14px; color:#A3AED0;">Harga Terakhir</div><div style="font-size:24px; font-weight:700; color:#2B3674;">Rp {price:,.0f}</div></div>""", unsafe_allow_html=True)
                except:
                    k1.markdown(f"""<div class="css-card" style="text-align:center;"><div style="font-size:14px; color:#A3AED0;">Harga Terakhir</div><div style="font-size:24px; font-weight:700; color:#2B3674;">N/A</div></div>""", unsafe_allow_html=True)
                
                # Metric 2: Flow
                try:
                    flow_val = float(last_row.get('Total_chg_Rp', 0))
                    flow_color = '#05CD99' if flow_val > 0 else '#EE5D50'
                    k2.markdown(f"""<div class="css-card" style="text-align:center;"><div style="font-size:14px; color:#A3AED0;">Flow {date_for_display}</div><div style="font-size:24px; font-weight:700; color: {flow_color};">{format_id_short(flow_val, True)}</div></div>""", unsafe_allow_html=True)
                except:
                    k2.markdown(f"""<div class="css-card" style="text-align:center;"><div style="font-size:14px; color:#A3AED0;">Flow {date_for_display}</div><div style="font-size:24px; font-weight:700; color: #A3AED0;">N/A</div></div>""", unsafe_allow_html=True)
                
                # Metric 3: Conviction Score (DENGAN ERROR HANDLING)
                try:
                    conviction_score, conviction_details = calculate_institutional_conviction(df, sel_stock)
                    
                    # Determine badge color
                    if conviction_score >= 80:
                        badge_color = "badge-high"
                        badge_text = "HIGH"
                    elif conviction_score >= 60:
                        badge_color = "badge-medium"
                        badge_text = "MEDIUM"
                    else:
                        badge_color = "badge-low"
                        badge_text = "LOW"
                    
                    k3.markdown(f"""<div class="css-card" style="text-align:center;"><div style="font-size:14px; color:#A3AED0;">Conviction Score</div><div style="font-size:24px; font-weight:700; color:#2B3674;">{conviction_score:.0f}</div><div><span class="{badge_color}">{badge_text}</span></div></div>""", unsafe_allow_html=True)
                except Exception as e:
                    # Fallback jika fungsi error
                    print(f"Error calculating conviction score: {e}")
                    k3.markdown(f"""<div class="css-card" style="text-align:center;"><div style="font-size:14px; color:#A3AED0;">Conviction Score</div><div style="font-size:24px; font-weight:700; color:#A3AED0;">N/A</div><div><span class="badge-low">ERROR</span></div></div>""", unsafe_allow_html=True)
                    conviction_score = 0
                
                # Metric 4: Pattern Detection (DENGAN ERROR HANDLING)
                try:
                    is_stealth, stealth_details = detect_stealth_accumulation(df, sel_stock)
                    sector = last_row.get('Sector', '-')
                    
                    k4.markdown(f"""<div class="css-card" style="text-align:center;"><div style="font-size:14px; color:#A3AED0;">Pattern Detection</div><div style="font-size:18px; font-weight:700; color:#2B3674;">{sector}</div><div style="font-size:12px; margin-top:5px;">{'🕵️ Stealth Detected' if is_stealth else 'Normal Pattern'}</div></div>""", unsafe_allow_html=True)
                except Exception as e:
                    print(f"Error in pattern detection: {e}")
                    k4.markdown(f"""<div class="css-card" style="text-align:center;"><div style="font-size:14px; color:#A3AED0;">Pattern Detection</div><div style="font-size:18px; font-weight:700; color:#2B3674;">-</div><div style="font-size:12px; margin-top:5px;">Error</div></div>""", unsafe_allow_html=True)

                # Sankey Chart dengan error handling
                st.markdown('<div class="css-card">', unsafe_allow_html=True)
                mode_sankey = st.radio("Mode Visualisasi:", ["Value (Rp)", "Volume (Lot)"], horizontal=True, label_visibility="collapsed")
                mode_key = 'Value' if 'Rp' in mode_sankey else 'Volume'
                
                try:
                    # Pilih tanggal untuk sankey
                    if not df_stock_month.empty:
                        sankey_date = df_stock_month.iloc[0]['Date']
                    elif not df_stock_all.empty:
                        sankey_date = df_stock_all['Date'].max()
                    else:
                        sankey_date = None
                    
                    if sankey_date is not None:
                        fig_sankey = create_sankey_chart(df, sel_stock, sankey_date, mode=mode_key)
                        if fig_sankey: 
                            st.plotly_chart(fig_sankey, use_container_width=True)
                        else: 
                            st.info("Pergerakan tidak cukup signifikan untuk Sankey.")
                    else:
                        st.info("Tidak ada data untuk visualisasi Sankey")
                except Exception as e:
                    st.error(f"Error creating Sankey chart: {e}")
                    st.info("Coba pilih saham atau bulan lain.")
                
                st.markdown('</div>', unsafe_allow_html=True)

                # --- [NEW] CHART HARGA VS FLOW KUMULATIF ---
                st.markdown('<div class="css-card"><div class="card-title">📈 Price Action vs Smart Money Flow</div>', unsafe_allow_html=True)
                
                # Prepare Data
                df_chart = df_stock_all.copy()
                
                # Hitung Kumulatif Flow Smart Money
                sm_cols_chart = get_available_columns(df_chart, SMART_MONEY_COLS)
                if sm_cols_chart:
                    df_chart['Smart_Money_Net'] = df_chart[sm_cols_chart].sum(axis=1)
                    df_chart['Cum_SM_Flow'] = df_chart['Smart_Money_Net'].cumsum()
                    
                    # Create Dual Axis Chart
                    fig_corr = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # 1. Price Line
                    fig_corr.add_trace(
                        go.Scatter(x=df_chart['Date'], y=df_chart['Price'], name="Price",
                                   line=dict(color='#2B3674', width=2)),
                        secondary_y=False
                    )
                    
                    # 2. Cumulative Flow Line
                    fig_corr.add_trace(
                        go.Scatter(x=df_chart['Date'], y=df_chart['Cum_SM_Flow'], name="Inst. Accumulation (Rp)",
                                   line=dict(color='#05CD99', width=2, dash='dot'), fill='tozeroy', fillcolor='rgba(5, 205, 153, 0.1)'),
                        secondary_y=True
                    )
                    
                    fig_corr.update_layout(
                        title=f"Korelasi Harga vs Akumulasi Institusi ({sel_stock})",
                        height=450,
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.1)
                    )
                    fig_corr = update_plotly_layout(fig_corr)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    st.caption("💡 **Tips:** Jika Garis Putus-putus (Flow) NAIK tapi Garis Biru (Harga) TURUN/FLAT -> **Divergence (Bullish)**")
                else:
                    st.warning("Data Smart Money tidak cukup untuk membuat chart.")
                st.markdown('</div>', unsafe_allow_html=True)

                # Monthly History dengan error handling
                st.markdown('<div class="css-card"><div class="card-title">📅 Monthly History</div>', unsafe_allow_html=True)
                try:
                    df_hist = calculate_monthly_change_table(df_stock_all)
                    if not df_hist.empty:
                        # Format kolom yang tersedia
                        available_hist_cols = [col for col in OWNERSHIP_CHG_VOL_COLS if col in df_hist.columns]
                        if available_hist_cols:
                            try:
                                styled_df = df_hist.style.format("{:,.0f}", subset=available_hist_cols)
                                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                            except:
                                st.dataframe(df_hist, use_container_width=True, hide_index=True)
                        else:
                            st.dataframe(df_hist, use_container_width=True, hide_index=True)
                    else:
                        st.info("Tidak ada data history")
                except Exception as e:
                    st.error(f"Error loading monthly history: {e}")
                    st.info("Data history tidak tersedia")
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Pilih saham untuk melihat detail")
        else:
            st.warning("Tidak ada data saham tersedia")
    else:
        st.warning("Data tidak tersedia")

# --- TAB 4: SCREENER ---
with tab4:
    if not df_filtered_month.empty and selected_month_str:
        st.markdown(f'<div class="css-card"><div class="card-title">🔍 Big Rotation Screener ({selected_month_str})</div>', unsafe_allow_html=True)
        
        # Gunakan kolom yang tersedia
        buyer_col = 'Top_Buyer_Val' if 'Top_Buyer_Val' in df_filtered_month.columns else 'Top_Buyer_Value_Rp'
        seller_col = 'Top_Seller_Val' if 'Top_Seller_Val' in df_filtered_month.columns else 'Top_Seller_Value_Rp'
        
        if buyer_col in df_filtered_month.columns and seller_col in df_filtered_month.columns:
            mask = (df_filtered_month[buyer_col].abs() >= min_rotation) | (df_filtered_month[seller_col].abs() >= min_rotation)
            df_scr = df_filtered_month[mask].copy()
            
            if not df_scr.empty:
                # Sort dan hitung metrics DENGAN ERROR HANDLING
                df_scr = df_scr.sort_values(buyer_col, ascending=False)
                
                # 🔥 PERBAIKAN: Gunakan try-except dalam apply
                def safe_calculate_conviction(code):
                    try:
                        score, _ = calculate_institutional_conviction(df, code)
                        return score
                    except:
                        return 0  # Default value jika error
                
                # 🔥 PERBAIKAN: Gunakan try-except untuk stealth detection
                def safe_detect_stealth(code):
                    try:
                        is_stealth, _ = detect_stealth_accumulation(df, code)
                        return is_stealth
                    except:
                        return False
                
                # Apply dengan fungsi yang aman
                df_scr['Conviction_Score'] = df_scr['Code'].apply(safe_calculate_conviction)
                df_scr['Is_Stealth'] = df_scr['Code'].apply(safe_detect_stealth)
                
                # Pilih kolom untuk display
                display_cols = ['Code', 'Sector', 'Conviction_Score', 'Is_Stealth', 
                               'Top_Buyer', buyer_col, 'Top_Seller', seller_col]
                available_display_cols = [col for col in display_cols if col in df_scr.columns]
                
                disp = df_scr[available_display_cols].copy()
                
                # Rename columns untuk display
                rename_dict = {
                    'Code': 'Code',
                    'Sector': 'Sector',
                    'Conviction_Score': 'Conviction',
                    'Is_Stealth': 'Stealth',
                    'Top_Buyer': 'Buyer',
                    buyer_col: 'Buy Val',
                    'Top_Seller': 'Seller',
                    seller_col: 'Sell Val'
                }
                
                disp = disp.rename(columns=rename_dict)
                
                # Pastikan semua kolom yang diperlukan ada
                expected_cols = ['Code', 'Sector', 'Conviction', 'Stealth', 'Buyer', 'Buy Val', 'Seller', 'Sell Val']
                for col in expected_cols:
                    if col not in disp.columns:
                        disp[col] = None
                
                # Style dataframe
                def style_conviction(val):
                    try:
                        val_float = float(val)
                        if val_float >= 80: 
                            return 'color: #0D9D58; font-weight: bold;'
                        elif val_float >= 60: 
                            return 'color: #FF9800; font-weight: bold;'
                        else: 
                            return 'color: #FF3B30; font-weight: bold;'
                    except:
                        return ''
                
                def style_flag(val):
                    try:
                        if bool(val): 
                            return 'background-color: #D6F5E3; color: #0D9D58; font-weight: bold; text-align: center;'
                        else: 
                            return ''
                    except:
                        return ''
                
                try:
                    # Format numeric columns
                    format_dict = {}
                    if 'Buy Val' in disp.columns:
                        format_dict['Buy Val'] = '{:,.0f}'
                    if 'Sell Val' in disp.columns:
                        format_dict['Sell Val'] = '{:,.0f}'
                    if 'Conviction' in disp.columns:
                        format_dict['Conviction'] = '{:.0f}'
                    
                    styled_df = disp.style
                    
                    if format_dict:
                        styled_df = styled_df.format(format_dict)
                    
                    # Apply styling
                    if 'Conviction' in disp.columns:
                        styled_df = styled_df.applymap(style_conviction, subset=['Conviction'])
                    
                    if 'Stealth' in disp.columns:
                        styled_df = styled_df.applymap(style_flag, subset=['Stealth'])
                    
                    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=600)
                except Exception as e:
                    # Fallback: tampilkan tanpa styling
                    st.error(f"Error styling dataframe: {e}")
                    st.dataframe(disp, use_container_width=True, hide_index=True)
            else:
                st.info(f"Tidak ada rotasi > Rp {min_rotation:,.0f} di bulan {selected_month_str}")
        else:
            st.warning("Kolom Top Buyer/Seller tidak tersedia")
    else:
        st.warning(f"Tidak ada data untuk bulan {selected_month_str}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: SMART SIGNALS ---
with tab5:
    if not df.empty:
        col_ai_filt, col_ai_res = st.columns([1, 3])
        
        with col_ai_filt:
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            win = st.slider("Window (Bulan)", 1, 6, 3)
            min_acc = st.number_input("Min Akumulasi (Miliar)", 5.0, 100.0, 5.0) * 1e9
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_ai_res:
            # Filter data 6 bulan terakhir
            date_cutoff = df['Date'].max() - pd.DateOffset(months=6)
            df_recent = df[df['Date'] >= date_cutoff]
            
            if not df_recent.empty:
                # 1. Calculate Signals
                df_sig = calculate_smart_money_signals(df_recent, win, min_acc)
                
                if not df_sig.empty:
                    st.markdown(f'<div class="css-card"><div class="card-title">💎 Smart Money Signals ({len(df_sig)})</div>', unsafe_allow_html=True)
                    
                    try:
                        df_display = df_sig.copy()
                        format_dict = {'Price': '{:,.0f}', 'Price Chg %': '{:.2f}%', 'Smart Money (Rp)': '{:,.0f}', 'Retail (Rp)': '{:,.0f}'}
                        available_format_cols = {k: v for k, v in format_dict.items() if k in df_display.columns}
                        
                        if available_format_cols:
                            styled = df_display.style.format(available_format_cols)
                            st.dataframe(styled, use_container_width=True, hide_index=True)
                        else:
                            st.dataframe(df_display, use_container_width=True, hide_index=True)
                    except:
                        st.dataframe(df_sig, use_container_width=True, hide_index=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Belum ada sinyal smart money yang terdeteksi dengan parameter ini")

                # 2. Clustering Analysis
                st.markdown('<div class="css-card"><div class="card-title">🤖 Smart Money Clustering Analysis</div>', unsafe_allow_html=True)
                
                try:
                    df_clusters = cluster_smart_money_patterns(df, n_clusters=4, sample_size=300)
                    
                    if not df_clusters.empty:
                        # --- 🔥 FIX: MEMBUAT KOLOM POSITIF UNTUK SIZE CHART ---
                        # Size tidak boleh negatif, jadi kita ambil nilai absolutnya
                        df_clusters['Viz_Size'] = df_clusters['Flow_Ratio'].abs()
                        # Kita fillna(0) untuk jaga-jaga agar tidak error
                        df_clusters['Viz_Size'] = df_clusters['Viz_Size'].fillna(0)

                        # Scatter Plot Visualization
                        st.markdown("#### 📈 Cluster Visualization")
                        
                        fig_cluster = px.scatter(
                            df_clusters, 
                            x='Smart_Flow_Miliar', 
                            y='Volatility', 
                            color='Cluster_Label',
                            size='Viz_Size', # <--- PAKE KOLOM BARU YANG SUDAH DI-ABS
                            hover_data=['Code', 'Sector', 'Smart_Flow_Miliar', 'Flow_Ratio'],
                            title="Smart Money Clusters: Flow vs Volatility",
                            color_discrete_map={
                                '🚀 Strong Accumulation': '#0D9D58',
                                '⚔️ Big Fight': '#FF9800',
                                '🕵️ Stealth Accumulation': '#0066CC',
                                '⚠️ Big Distribution': '#FF3B30',
                                '📊 Sideways/Retail': '#A3AED0'
                            }
                        )
                        
                        # Add reference lines
                        fig_cluster.add_hline(y=df_clusters['Volatility'].mean(), line_dash="dash", line_color="gray",
                                            annotation_text="Avg Volatility", annotation_position="bottom right")
                        fig_cluster.add_vline(x=0, line_dash="dash", line_color="gray")
                        
                        fig_cluster.update_layout(
                            height=500,
                            xaxis_title="Smart Money Flow (Miliar Rp)",
                            yaxis_title="Volatility (Std Dev / Mean)",
                            hovermode='closest'
                        )
                        
                        fig_cluster = update_plotly_layout(fig_cluster)
                        st.plotly_chart(fig_cluster, use_container_width=True)

                        # Tampilkan Cluster Distribution (Icon Boxes)
                        st.markdown("#### 📊 Cluster Distribution")
                        cluster_counts = df_clusters['Cluster_Label'].value_counts()
                        
                        col_dist1, col_dist2, col_dist3, col_dist4 = st.columns(4)
                        cols = [col_dist1, col_dist2, col_dist3, col_dist4]
                        
                        for idx, (label, count) in enumerate(cluster_counts.items()):
                            if idx < len(cols):
                                with cols[idx]:
                                    if "Strong" in label: icon, color = "🚀", "#0D9D58"
                                    elif "Fight" in label: icon, color = "⚔️", "#FF9800"
                                    elif "Stealth" in label: icon, color = "🕵️", "#0066CC"
                                    elif "Distribution" in label: icon, color = "⚠️", "#FF3B30"
                                    else: icon, color = "📊", "#A3AED0"
                                    
                                    st.markdown(f"""
                                        <div style="text-align: center; padding: 10px; background-color: {color}10; border-radius: 10px; border-left: 4px solid {color};">
                                            <div style="font-size: 24px;">{icon}</div>
                                            <div style="font-size: 12px; color: {color}; font-weight: bold;">{label}</div>
                                            <div style="font-size: 20px; font-weight: bold;">{count}</div>
                                        </div>
                                    """, unsafe_allow_html=True)

                    else:
                        st.info("Data clustering tidak cukup.")

                except Exception as e:
                    st.error(f"Error in clustering visual: {e}")

            else:
                st.info("Tidak ada data recent untuk analisis")
    else:
        st.warning("Data tidak tersedia")

# --- TAB 6: TOP MOVERS ---
with tab6:
    if not df_filtered_month.empty and selected_month_str:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">🔥 Top Movers ({selected_month_str})</div>', unsafe_allow_html=True)
        
        df_hot = get_significant_movements(df_filtered_month, threshold_rp=threshold_rp, threshold_pct=1.0)
        
        if not df_hot.empty:
            # Format dataframe
            try:
                format_cols = ['Price', 'Total Flow (Rp)', 'Net Flow (Rp)']
                available_format_cols = [col for col in format_cols if col in df_hot.columns]
                
                if available_format_cols:
                    format_dict = {col: '{:,.0f}' for col in available_format_cols}
                    styled_hot = df_hot.style.format(format_dict)
                    st.dataframe(styled_hot, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(df_hot, use_container_width=True, hide_index=True)
            except:
                st.dataframe(df_hot, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada pergerakan signifikan yang terdeteksi")
    else:
        st.warning(f"Tidak ada data untuk bulan {selected_month_str}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 7: INSTITUTIONAL INTELLIGENCE ---
with tab7:
    if not df.empty:
        st.markdown('<div class="header-banner" style="margin-bottom:20px; padding:20px;"><div class="header-title">🎯 Institutional Intelligence Engine</div></div>', unsafe_allow_html=True)
        
        # 1. High Conviction Stocks
        st.markdown('<div class="css-card"><div class="card-title">🏆 High Conviction Stocks</div>', unsafe_allow_html=True)
        
        # Filter settings
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            conv_thresh = st.slider("Minimum Conviction Score", 50, 100, conviction_threshold, 5,
                                  help="Filter saham dengan score conviction minimal")
        with col_filter2:
            min_flow_filter = st.number_input("Min Institutional Flow (Rp Miliar)", 
                                            value=5.0, step=1.0, format="%.1f") * 1e9
        
        try:
            with st.spinner("🔍 Scanning high conviction stocks..."):
                # Panggil fungsi dengan error handling
                df_high_conviction = scan_high_conviction_stocks(df, min_score=conv_thresh, min_flow=min_flow_filter)
            
            if not df_high_conviction.empty:
                st.success(f"✅ Ditemukan {len(df_high_conviction)} saham dengan conviction ≥ {conv_thresh}")
                
                # Tampilkan metrics summary
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    avg_score = df_high_conviction['Conviction_Score'].mean()
                    st.metric("Rata-rata Score", f"{avg_score:.1f}")
                with col_sum2:
                    stealth_count = df_high_conviction['Is_Stealth'].sum() if 'Is_Stealth' in df_high_conviction.columns else 0
                    st.metric("Stealth Patterns", f"{stealth_count}")
                with col_sum3:
                    coord_count = df_high_conviction['Is_Coordinated'].sum() if 'Is_Coordinated' in df_high_conviction.columns else 0
                    st.metric("Coordinated Moves", f"{coord_count}")
                
                # Prepare data for display
                df_display = df_high_conviction.copy()
                
                # Format columns
                if 'Price' in df_display.columns:
                    df_display['Price'] = df_display['Price'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else "N/A")
                
                if 'Institutional_Flow' in df_display.columns:
                    df_display['Institutional_Flow_Formatted'] = df_display['Institutional_Flow'].apply(
                        lambda x: format_id_short(x, True) if pd.notna(x) else "N/A"
                    )
                
                if 'Price_Chg_%' in df_display.columns:
                    df_display['Price_Chg_%'] = df_display['Price_Chg_%'].apply(
                        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
                    )
                
                # Select and rename columns for display
                display_map = {
                    'Code': 'Kode',
                    'Sector': 'Sektor',
                    'Price': 'Harga',
                    'Price_Chg_%': 'Perubahan %',
                    'Conviction_Score': 'Conviction',
                    'Institutional_Flow_Formatted': 'Flow Institusi',
                    'Is_Stealth': 'Stealth?',
                    'Is_Coordinated': 'Koordinasi?'
                }
                
                # Filter available columns
                available_cols = {}
                for orig, new in display_map.items():
                    if orig in df_display.columns:
                        available_cols[orig] = new
                
                if available_cols:
                    df_display = df_display[list(available_cols.keys())]
                    df_display = df_display.rename(columns=available_cols)
                    
                    # Style the dataframe
                    def highlight_conviction(val):
                        try:
                            val_float = float(val)
                            if val_float >= 90:
                                return 'background-color: #D6F5E3; color: #0D9D58; font-weight: bold;'
                            elif val_float >= 80:
                                return 'background-color: #FFF4E5; color: #FF9800; font-weight: bold;'
                            elif val_float >= 70:
                                return 'background-color: #FFE5E5; color: #FF3B30; font-weight: bold;'
                        except:
                            pass
                        return ''
                    
                    def highlight_bool(val):
                        if isinstance(val, bool):
                            if val:
                                return 'background-color: #D6F5E3; color: #0D9D58; font-weight: bold; text-align: center;'
                            else:
                                return 'text-align: center;'
                        return ''
                    
                    # Apply styling
                    styled_df = df_display.style
                    
                    if 'Conviction' in df_display.columns:
                        styled_df = styled_df.applymap(highlight_conviction, subset=['Conviction'])
                    
                    if 'Stealth?' in df_display.columns:
                        styled_df = styled_df.applymap(highlight_bool, subset=['Stealth?'])
                    
                    if 'Koordinasi?' in df_display.columns:
                        styled_df = styled_df.applymap(highlight_bool, subset=['Koordinasi?'])
                    
                    # Display the dataframe
                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        hide_index=True,
                        height=min(400, len(df_display) * 35 + 38)
                    )
                    
                    # Download button
                    csv = df_high_conviction.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download High Conviction Stocks",
                        data=csv,
                        file_name=f"high_conviction_stocks_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("Tidak ada kolom yang tersedia untuk ditampilkan")
                    
            else:
                st.info(f"📭 Tidak ditemukan saham dengan conviction score ≥ {conv_thresh}")
                st.caption("Coba turunkan threshold atau filter yang lebih longgar")
                
        except Exception as e:
            st.error(f"❌ Error scanning high conviction stocks: {str(e)}")
            st.info("⚠️ Coba refresh data atau turunkan threshold")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 2. Smart Money Clustering
        st.markdown('<div class="css-card"><div class="card-title">🤖 Smart Money Clustering Analysis</div>', unsafe_allow_html=True)
        
        col_clust1, col_clust2 = st.columns(2)
        with col_clust1:
            n_clusters = st.selectbox("Number of Clusters", [3, 4, 5, 6], index=1, 
                                    help="Jumlah cluster untuk grouping pattern", key="clust_n_tab7") # Tambah unique key
        with col_clust2:
            sample_size = st.slider("Sample Size", 50, 500, 200, 50, 
                                    help="Jumlah saham yang dianalisis (untuk performance)", key="clust_sample_tab7") # Tambah unique key
        
        try:
            with st.spinner("🤖 Processing clustering analysis..."):
                df_clusters = cluster_smart_money_patterns(df, n_clusters=n_clusters, sample_size=sample_size)
            
            if not df_clusters.empty:
                # --- 🔥 FIX CRITICAL: SIZE CHART TIDAK BOLEH NEGATIF ---
                # Kita buat kolom baru 'Viz_Size' yang berisi nilai mutlak (absolut) dari Flow_Ratio
                df_clusters['Viz_Size'] = df_clusters['Flow_Ratio'].abs()
                df_clusters['Viz_Size'] = df_clusters['Viz_Size'].fillna(0) # Handle NaN

                # Display cluster distribution
                st.markdown("#### 📊 Cluster Distribution")
                cluster_counts = df_clusters['Cluster_Label'].value_counts()
                
                col_dist1, col_dist2, col_dist3, col_dist4 = st.columns(4)
                cols = [col_dist1, col_dist2, col_dist3, col_dist4]
                
                for idx, (label, count) in enumerate(cluster_counts.items()):
                    if idx < len(cols):
                        with cols[idx]:
                            if "Strong" in label: icon, color = "🚀", "#0D9D58"
                            elif "Fight" in label: icon, color = "⚔️", "#FF9800"
                            elif "Stealth" in label: icon, color = "🕵️", "#0066CC"
                            elif "Distribution" in label: icon, color = "⚠️", "#FF3B30"
                            else: icon, color = "📊", "#A3AED0"
                            
                            st.markdown(f"""
                                <div style="text-align: center; padding: 10px; background-color: {color}10; border-radius: 10px; border-left: 4px solid {color};">
                                    <div style="font-size: 24px;">{icon}</div>
                                    <div style="font-size: 12px; color: {color}; font-weight: bold;">{label}</div>
                                    <div style="font-size: 20px; font-weight: bold;">{count}</div>
                                </div>
                            """, unsafe_allow_html=True)
                
                # Scatter plot visualization
                st.markdown("#### 📈 Cluster Visualization")
                
                fig_cluster = px.scatter(
                    df_clusters, 
                    x='Smart_Flow_Miliar', 
                    y='Volatility', 
                    color='Cluster_Label',
                    size='Viz_Size', # <--- SUDAH DIPERBAIKI (PAKE NILAI MUTLAK)
                    hover_data=['Code', 'Sector', 'Smart_Flow_Miliar', 'Flow_Ratio'],
                    title="Smart Money Clusters: Flow vs Volatility",
                    color_discrete_map={
                        '🚀 Strong Accumulation': '#0D9D58',
                        '⚔️ Big Fight': '#FF9800',
                        '🕵️ Stealth Accumulation': '#0066CC',
                        '⚠️ Big Distribution': '#FF3B30',
                        '📊 Sideways/Retail': '#A3AED0'
                    }
                )
                
                # Add reference lines
                fig_cluster.add_hline(y=df_clusters['Volatility'].mean(), line_dash="dash", line_color="gray",
                                    annotation_text="Avg Volatility", annotation_position="bottom right")
                fig_cluster.add_vline(x=0, line_dash="dash", line_color="gray")
                
                fig_cluster.update_layout(
                    height=500,
                    xaxis_title="Smart Money Flow (Miliar Rp)",
                    yaxis_title="Volatility (Std Dev / Mean)",
                    hovermode='closest'
                )
                
                fig_cluster = update_plotly_layout(fig_cluster)
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Cluster details table
                st.markdown("#### 📋 Cluster Details")
                
                selected_clusters = st.multiselect(
                    "Filter Clusters:",
                    options=df_clusters['Cluster_Label'].unique(),
                    default=df_clusters['Cluster_Label'].unique()[:2],
                    key="clust_filter_tab7" # Unique key agar tidak bentrok dengan Tab 5
                )
                
                if selected_clusters:
                    filtered_clusters = df_clusters[df_clusters['Cluster_Label'].isin(selected_clusters)]
                    
                    # Format for display
                    cluster_display = filtered_clusters[['Code', 'Sector', 'Cluster_Label', 
                                                       'Smart_Flow_Miliar', 'Flow_Ratio', 'Volatility']].copy()
                    
                    cluster_display['Smart_Flow_Miliar'] = cluster_display['Smart_Flow_Miliar'].apply(lambda x: f"{x:,.1f}")
                    cluster_display['Flow_Ratio'] = cluster_display['Flow_Ratio'].apply(lambda x: f"{x:.2f}")
                    cluster_display['Volatility'] = cluster_display['Volatility'].apply(lambda x: f"{x:.3f}")
                    
                    cluster_display = cluster_display.rename(columns={
                        'Code': 'Kode',
                        'Sector': 'Sektor',
                        'Cluster_Label': 'Cluster',
                        'Smart_Flow_Miliar': 'Flow (Miliar Rp)',
                        'Flow_Ratio': 'Flow Ratio',
                        'Volatility': 'Volatilitas'
                    })
                    
                    st.dataframe(cluster_display, use_container_width=True, hide_index=True)
                    
                    # Download cluster data
                    csv_clusters = df_clusters.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Cluster Data",
                        data=csv_clusters,
                        file_name=f"smart_money_clusters_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="btn_dl_clust_tab7" # Unique key
                    )
                else:
                    st.info("Pilih cluster untuk melihat detail")
                    
            else:
                st.info("📭 Tidak ada data yang cukup untuk clustering analysis")
                
        except Exception as e:
            st.error(f"❌ Error in clustering analysis: {str(e)}")
            st.info("⚠️ Clustering tidak tersedia saat ini. Coba dengan parameter berbeda.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.warning("📭 Data tidak tersedia untuk institutional intelligence analysis")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #A3AED0;'>KSEI Bandarmology PRO | Institutional Intelligence Platform | Updated for new column format</div>", unsafe_allow_html=True)
