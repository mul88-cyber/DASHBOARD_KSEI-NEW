# ==============================================================================
# 📦 1) IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import traceback
from datetime import datetime, timedelta

# Import library Google
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ 2) KONFIGURASI DASHBOARD & G-DRIVE
# ==============================================================================
st.set_page_config(
    page_title="🌊 Dashboard Bandarmology KSEI (Bulanan)",
    layout="wide",
    page_icon="🌊"
)

# Tambahkan custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #0e1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3746;
    }
    .stMetric label {
        font-weight: bold;
        color: #83c9ff;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .streamlit-expanderHeader {
        background-color: rgba(28, 131, 225, 0.1);
        border-radius: 5px;
    }
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

# --- KONFIGURASI KELOMPOK PEMAIN (BANDARMOLOGY) ---
SMART_MONEY_COLS = [
    'Foreign IS_chg_Rp', 'Foreign IB_chg_Rp', 'Foreign PF_chg_Rp', 
    'Local IS_chg_Rp', 'Local PF_chg_Rp', 'Local MF_chg_Rp', 'Local IB_chg_Rp'
]
RETAIL_COLS = ['Local ID_chg_Rp']

# ==============================================================================
# 📦 3) FUNGSI MEMUAT DATA (via SERVICE ACCOUNT)
# ==============================================================================
def get_gdrive_service():
    """Membuat service untuk Google Drive API."""
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None
    except KeyError:
        msg = "❌ Gagal otentikasi: 'st.secrets' tidak menemukan key [gcp_service_account]."
        return None, msg
    except Exception as e:
        msg = f"❌ Gagal otentikasi Google Drive: {e}."
        return None, msg

@st.cache_data(ttl=3600, show_spinner="🔄 Memuat data dari Google Drive...")
def load_data():
    """Mencari file KSEI, men-download, membersihkan, dan filter data 2025 ke atas."""
    service, error_msg = get_gdrive_service()
    if error_msg:
        return pd.DataFrame(), error_msg, "error"

    try:
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(
            q=query, fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1
        ).execute()
        items = results.get('files', [])

        if not items:
            msg = f"❌ File '{FILE_NAME}' tidak ditemukan di folder GDrive."
            return pd.DataFrame(), msg, "error"

        file_id = items[0]['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)

        df = pd.read_csv(fh, dtype=object)
        
        # Validasi kolom minimal
        required_cols = ['Date', 'Code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            msg = f"❌ Kolom penting hilang: {missing_cols}"
            return pd.DataFrame(), msg, "error"

        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # --- [FILTER UTAMA] HANYA DATA 2025 KE ATAS ---
        df = df[df['Date'].dt.year >= 2025].copy()
        
        if df.empty:
             msg = "❌ Data tahun 2025 ke atas tidak ditemukan di file."
             return pd.DataFrame(), msg, "error"

        if 'Sector' in df.columns:
            df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
            df['Sector'] = 'Others'

        cols_to_numeric = [
            'Price', 'Price_Chg %', 'Free Float', 'Total_Local', 'Total_Foreign',
            'Top_Buyer_Vol', 'Top_Seller_Vol', 'Sec. Num',
            'Top_Buyer_Value_Rp', 'Top_Seller_Value_Rp'
        ] + OWNERSHIP_COLS + OWNERSHIP_CHG_COLS + OWNERSHIP_CHG_RP_COLS

        for col in cols_to_numeric:
            if col in df.columns:
                cleaned_col = df[col].astype(str).str.strip()
                cleaned_col = cleaned_col.str.replace(',', '', regex=False)
                cleaned_col = cleaned_col.str.replace('[^\d.-]', '', regex=True)
                df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0)

        df = df.dropna(subset=['Date', 'Code'])

        # Cek duplikat
        dupes = df.duplicated(subset=['Date', 'Code']).sum()
        if dupes > 0:
            df = df.drop_duplicates(subset=['Date', 'Code'], keep='last')

        local_chg_rp_cols = [col for col in OWNERSHIP_CHG_RP_COLS if 'Local' in col and col in df.columns]
        foreign_chg_rp_cols = [col for col in OWNERSHIP_CHG_RP_COLS if 'Foreign' in col and col in df.columns]
        df['Total_Local_chg_Rp'] = df[local_chg_rp_cols].sum(axis=1)
        df['Total_Foreign_chg_Rp'] = df[foreign_chg_rp_cols].sum(axis=1)
        df['Total_chg_Rp'] = df['Total_Local_chg_Rp'] + df['Total_Foreign_chg_Rp']

        msg = f"✅ Data KSEI (2025-Now) berhasil dimuat: {len(df)} baris, {df['Date'].nunique()} bulan (file ID: {file_id})."
        return df, msg, "success"

    except Exception as e:
        msg = f"❌ Terjadi error saat memuat data KSEI: {e}."
        return pd.DataFrame(), msg, "error"

def validate_data(df):
    """Validasi integrity data."""
    if df.empty:
        return False, "DataFrame kosong"
    
    required_cols = ['Date', 'Code', 'Price']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, f"Kolom penting hilang: {missing}"
    
    # Cek nilai null
    null_counts = df[['Date', 'Code']].isnull().sum()
    if null_counts.any():
        return False, f"Ada nilai null: {null_counts[null_counts > 0].to_dict()}"
    
    return True, "Data valid"

# ==============================================================================
# 🛠️ 4) FUNGSI KALKULASI & FORMATTING (UPDATED)
# ==============================================================================

def format_id_short(value, is_currency=False):
    """Format angka ke Juta (Jt), Milyar (M), Triliun (T)."""
    if pd.isna(value) or value == 0:
        return "0"
    
    val_abs = abs(value)
    suffix = ""
    divisor = 1
    
    if val_abs >= 1_000_000_000_000:
        suffix = " T"
        divisor = 1_000_000_000_000
    elif val_abs >= 1_000_000_000:
        suffix = " M"
        divisor = 1_000_000_000
    elif val_abs >= 1_000_000:
        suffix = " Jt"
        divisor = 1_000_000
    else:
        return f"{value:,.0f}" # Angka kecil biasa
    
    formatted_num = f"{value/divisor:.1f}" # 1 desimal
    # Hapus .0 di belakang (misal 5.0 M -> 5 M)
    if formatted_num.endswith(".0"):
        formatted_num = formatted_num[:-2]
        
    prefix = "Rp " if is_currency else ""
    return f"{prefix}{formatted_num}{suffix}"

@st.cache_data
def calculate_macro_flow(df_filtered):
    """(TAB 1) Menghitung total aliran dana per kategori di seluruh market."""
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
    """(TAB 2) Menghitung aliran dana bersih kategori tertentu per sektor."""
    category_chg_col = f"{selected_category}_chg_Rp"
    if category_chg_col not in df_filtered.columns:
        return pd.DataFrame(), f"Kolom '{category_chg_col}' tidak ditemukan."
        
    sector_flow = df_filtered.groupby('Sector')[category_chg_col].sum().reset_index()
    sector_flow.columns = ['Sector', 'Net Flow (Rp)']
    sector_flow = sector_flow.sort_values(by='Net Flow (Rp)', ascending=False)
    return sector_flow, None

@st.cache_data
def get_stock_ownership_state(df, stock_code):
    """(TAB 3) Mengambil data kepemilikan TERBARU untuk 1 saham."""
    df_stock = df[df['Code'] == stock_code]
    if df_stock.empty:
        return pd.DataFrame(), pd.Series(dtype='object')
    latest_row = df_stock.sort_values('Date').iloc[-1]
    
    # Ambil kolom kepemilikan saja
    df_state = latest_row[OWNERSHIP_COLS].reset_index()
    df_state.columns = ['Kategori', 'Jumlah Saham']
    
    return df_state, latest_row

@st.cache_data
def calculate_monthly_change_table(df_stock):
    """(TAB 3 Table) Menghitung perubahan bulanan (Volume)."""
    df_display = df_stock.sort_values('Date', ascending=False).copy()
    cols = ['Date'] + OWNERSHIP_CHG_COLS
    df_res = df_display[cols].copy()
    df_res.rename(columns={'Date': 'Bulan'}, inplace=True)
    df_res['Bulan'] = df_res['Bulan'].dt.strftime('%b %Y')
    return df_res

@st.cache_data
def calculate_monthly_sector_flow(df_filtered):
    """(TAB 4 Chart) Menghitung total aliran dana bulanan per sektor."""
    df_temp = df_filtered.set_index('Date')
    monthly_flow = df_temp.groupby('Sector')['Total_chg_Rp'].resample('MS').sum().reset_index()
    monthly_flow.columns = ['Sector', 'Month', 'Net Flow (Rp)']
    return monthly_flow, None

@st.cache_data
def calculate_smart_money_signals(df_year, window_periods=3, min_acc_threshold=5_000_000_000):
    """(TAB 5) Algoritma Mencari Saham Potensial."""
    if df_year.empty: return pd.DataFrame()
    codes = df_year['Code'].unique()
    results = []
    for code in codes:
        df_stock = df_year[df_year['Code'] == code].sort_values('Date')
        df_window = df_stock.tail(window_periods)
        if df_window.empty: continue
        last_price = df_window.iloc[-1]['Price']
        start_price = df_window.iloc[0]['Price']
        price_chg_pct = 0
        if start_price > 0:
            price_chg_pct = ((last_price - start_price) / start_price) * 100
        valid_sm_cols = [c for c in SMART_MONEY_COLS if c in df_stock.columns]
        valid_ret_cols = [c for c in RETAIL_COLS if c in df_stock.columns]
        sm_flow_sum = df_window[valid_sm_cols].sum().sum()
        retail_flow_sum = df_window[valid_ret_cols].sum().sum()
        status = "Netral"; score = 0
        if sm_flow_sum > min_acc_threshold and retail_flow_sum < 0: status = "🔥 Big Accumulation"; score = 100
        elif sm_flow_sum > (min_acc_threshold/2) and price_chg_pct <= 3: status = "💎 Divergence (Collect)"; score = 80
        elif sm_flow_sum < -min_acc_threshold and retail_flow_sum > 0: status = "⚠️ Distribution"; score = -50
        if abs(score) >= 50:
            results.append({'Code': code, 'Sector': df_window.iloc[-1].get('Sector', 'N/A'), 'Price': last_price, 'Price Chg (Window)%': price_chg_pct, 'Smart Money Flow (Rp)': sm_flow_sum, 'Retail Flow (Rp)': retail_flow_sum, 'Signal': status, 'Score': score})
    df_res = pd.DataFrame(results)
    if not df_res.empty: df_res = df_res.sort_values(by='Smart Money Flow (Rp)', ascending=False)
    return df_res

@st.cache_data
def filter_by_month(df, selected_month):
    """Filter data berdasarkan bulan tertentu."""
    if not isinstance(selected_month, str):
        selected_month = selected_month.strftime('%Y-%m')
    
    # Filter bulan (format YYYY-MM)
    df_filtered = df[df['Date'].dt.strftime('%Y-%m') == selected_month]
    return df_filtered

@st.cache_data
def get_significant_movements(df_month, threshold_rp=10_000_000_000, threshold_pct=5):
    """Mendapatkan saham dengan pergerakan signifikan di bulan tertentu."""
    if df_month.empty:
        return pd.DataFrame()
    
    results = []
    for code in df_month['Code'].unique():
        df_stock = df_month[df_month['Code'] == code].iloc[0]  # Ambil row pertama (hanya ada 1 per bulan)
        
        # Hitung total flow (absolut)
        total_flow_abs = 0
        for col in OWNERSHIP_CHG_RP_COLS:
            if col in df_stock:
                total_flow_abs += abs(df_stock[col])
        
        # Hitung flow net
        total_flow_net = df_stock.get('Total_chg_Rp', 0)
        
        # Hitung % perubahan dari total kepemilikan
        total_shares = df_stock.get('Sec. Num', 1)
        flow_pct = (total_flow_abs / total_shares * 100) if total_shares > 0 else 0
        
        # Cek apakah signifikan
        is_significant = (total_flow_abs >= threshold_rp) or (flow_pct >= threshold_pct)
        
        if is_significant:
            # Tentukan arah
            direction = "NET BUY" if total_flow_net > 0 else "NET SELL" if total_flow_net < 0 else "NEUTRAL"
            
            # Identifikasi top buyer/seller kategori
            buyer_cats = []
            seller_cats = []
            
            for col in OWNERSHIP_CHG_RP_COLS:
                if col in df_stock:
                    val = df_stock[col]
                    if val > threshold_rp * 0.1:  # Minimal 10% dari threshold
                        buyer_cats.append(col.replace('_chg_Rp', ''))
                    elif val < -threshold_rp * 0.1:
                        seller_cats.append(col.replace('_chg_Rp', ''))
            
            results.append({
                'Code': code,
                'Sector': df_stock.get('Sector', 'N/A'),
                'Price': df_stock.get('Price', 0),
                'Total Flow (Rp)': total_flow_abs,
                'Net Flow (Rp)': total_flow_net,
                'Flow % of Shares': flow_pct,
                'Direction': direction,
                'Top Buyers': ', '.join(buyer_cats[:3]) if buyer_cats else '-',
                'Top Sellers': ', '.join(seller_cats[:3]) if seller_cats else '-',
                'Top_Buyer': df_stock.get('Top_Buyer', '-'),
                'Top_Seller': df_stock.get('Top_Seller', '-')
            })
    
    df_result = pd.DataFrame(results)
    if not df_result.empty:
        df_result = df_result.sort_values('Total Flow (Rp)', ascending=False)
    
    return df_result

# --- FUNGSI SANKEY CHART (UPDATED WITH LABELS) ---
def create_sankey_chart(df, stock_code, selected_date, mode='Volume'):
    """Membuat diagram Sankey dengan label angka (Jt/M/T)."""
    
    # 1. Filter Data
    row = df[(df['Code'] == stock_code) & (df['Date'] == selected_date)]
    if row.empty:
        # Fallback ke tanggal terbaru untuk saham ini
        stock_dates = df[df['Code'] == stock_code]['Date']
        if not stock_dates.empty:
            selected_date = stock_dates.max()
            row = df[(df['Code'] == stock_code) & (df['Date'] == selected_date)]
        else:
            return None
    
    row = row.iloc[0]
    
    # 2. Tentukan Kolom & Satuan
    cols_to_use = OWNERSHIP_CHG_COLS if mode == 'Volume' else OWNERSHIP_CHG_RP_COLS
    is_rp = (mode == 'Value')
    
    # 3. Pisahkan Seller & Buyer
    sellers = []; buyers = []; total_vol = 0
    
    for col in cols_to_use:
        val = row[col] if col in row else 0
        cat_name = col.replace('_chg_Rp', '').replace('_chg', '')
        
        if val != 0:
            formatted_val = format_id_short(abs(val), is_currency=is_rp)
            # Label Node: "Foreign IB\n(2.5 M)"
            label_node = f"{cat_name}\n({formatted_val})"
            
            if val < 0:
                sellers.append({'label': label_node, 'value': abs(val), 'raw_name': cat_name})
                total_vol += abs(val)
            elif val > 0:
                buyers.append({'label': label_node, 'value': val, 'raw_name': cat_name})
            
    if not sellers and not buyers: 
        return None
    
    # 4. Bangun Node
    # Market Node Label
    market_fmt = format_id_short(total_vol, is_currency=is_rp)
    labels = [f"MARKET\n({market_fmt})"]
    colors = ["lightgrey"]
    
    source = []; target = []; values = []; link_colors = []
    
    # Sellers -> Market
    for s in sellers:
        labels.append(s['label'])
        current_idx = len(labels) - 1
        colors.append("#ff6b6b")
        source.append(current_idx); target.append(0); values.append(s['value'])
        link_colors.append("rgba(255, 107, 107, 0.4)")
        
    # Market -> Buyers
    for b in buyers:
        labels.append(b['label'])
        current_idx = len(labels) - 1
        colors.append("#51cf66")
        source.append(0); target.append(current_idx); values.append(b['value'])
        link_colors.append("rgba(81, 207, 102, 0.4)")
        
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15, thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = labels, color = colors
        ),
        link = dict(
            source = source, target = target, value = values, color = link_colors,
            hovertemplate='%{source.label} -> %{target.label}<br>Value: %{value:,.0f}<extra></extra>'
        ))])

    title_text = f"Arus Dana {stock_code} ({selected_date.strftime('%b %Y')}) - {mode}"
    fig.update_layout(title_text=title_text, font_size=11, height=500)
    return fig

def highlight_max_min(s):
    """Highlight max dan min di dataframe."""
    s_numeric = pd.to_numeric(s, errors='coerce')
    max_val = s_numeric[s_numeric > 0].max()
    min_val = s_numeric[s_numeric < 0].min()
    colors = []
    for val in s_numeric:
        if pd.notna(val):
            if val == max_val and val > 0: colors.append('background-color: #90EE90')
            elif val == min_val and val < 0: colors.append('background-color: #FFB6C1')
            else: colors.append('')
        else: colors.append('')
    return colors

# ==============================================================================
# 💎 5) LAYOUT UTAMA
# ==============================================================================
st.title("🌊 Dashboard Bandarmology KSEI (Data Bulanan)")
st.caption("Analisis khusus data 2025 ke atas.")

try:
    # Load data dengan progress indicator
    with st.spinner("🔄 Memuat data dari Google Drive..."):
        df, status_msg, status_level = load_data()
    
    if status_level == "success":
        st.toast(status_msg, icon="✅")
    elif status_level == "error":
        st.error(status_msg)
        st.stop()
    
    # Validasi data
    is_valid, valid_msg = validate_data(df)
    if not is_valid:
        st.warning(f"⚠️ {valid_msg}")
        
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    with st.expander("🔧 Error Details"):
        st.code(traceback.format_exc())
    st.stop()

# ==============================================================================
# 🧭 SIDEBAR - IMPROVED
# ==============================================================================
with st.sidebar:
    st.header("🎛️ Filter & Navigasi")
    
    # Refresh dan Clear Cache
    col_refresh, col_clear = st.columns(2)
    with col_refresh:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_clear:
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    st.divider()
    
    # Info Data
    st.info(f"""
    📅 Data Range:  
    **{df['Date'].dt.date.min()}** s/d **{df['Date'].dt.date.max()}**  
    
    📊 Statistik:  
    • {len(df):,} baris data  
    • {df['Code'].nunique()} saham  
    • {df['Sector'].nunique()} sektor  
    • {df['Date'].nunique()} bulan
    """)
    
    st.divider()
    
    # 🔥 NEW: FILTER BY MONTH untuk semua tab
    st.header("📅 Filter Berdasarkan Bulan")
    
    # Dapatkan bulan unik
    available_months = sorted(df['Date'].dt.strftime('%Y-%m').unique(), reverse=True)
    selected_month_str = st.selectbox(
        "Pilih Bulan:",
        options=available_months,
        index=0,
        help="Filter data untuk bulan tertentu"
    )
    
    # Konversi ke datetime untuk filter
    selected_month = pd.to_datetime(selected_month_str + '-01')
    
    # Filter data berdasarkan bulan yang dipilih
    df_filtered_month = filter_by_month(df, selected_month_str)
    
    # Threshold untuk significant movements
    st.subheader("⚡ Threshold Pergerakan Signifikan")
    threshold_rp = st.number_input(
        "Min. Value Flow (Rp)",
        value=10_000_000_000,
        step=1_000_000_000,
        format="%d",
        help="Nilai minimum flow untuk dianggap signifikan"
    )
    
    threshold_pct = st.slider(
        "Min. % dari Total Saham",
        min_value=0.1,
        max_value=20.0,
        value=5.0,
        step=0.1,
        help="Persentase minimum flow dari total saham"
    )
    
    # Filter Screener (Tab 4)
    st.divider()
    st.header("🔍 Filter Screener (Tab 4)")
    min_rotation_value = st.number_input(
        "Min. Value Rotasi (Rp)", 
        value=1_000_000_000, 
        step=1_000_000_000, 
        format="%d"
    )
    
    # Export Data
    st.divider()
    st.header("📤 Export Data")
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full CSV",
            data=csv,
            file_name=f"ksei_bandarmology_full_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Panduan Cepat
    with st.expander("📖 Panduan Cepat", expanded=False):
        st.markdown("""
        **Tab 1**: Aliran dana market secara agregat  
        **Tab 2**: Rotasi sektor per kategori investor  
        **Tab 3**: Analisis mendalam per saham  
        **Tab 4**: Screener berdasarkan volume rotasi  
        **Tab 5**: Sinyal smart money vs retail  
        **Tab 6**: 🔥 **NEW** Pergerakan Signifikan per Bulan
        """)

# ==============================================================================
# 📑 TABS VISUALISASI - DITAMBAH TAB BARU
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌊 **Makro (Market)**",
    "📊 **Analisis Sektor**",
    "📈 **Analisa Individual**", 
    "🔍 **Screener Rotasi**",
    "💎 **Sinyal Potensial (AI)**",
    "🔥 **Pergerakan Signifikan**"  # TAB BARU
])

# --- TAB 1: MAKRO ---
with tab1:
    st.subheader("Peta Aliran Dana Market (2025)")
    df_net_flow, df_cum_flow = calculate_macro_flow(df)
    
    # Tampilkan bulan yang aktif
    st.info(f"**Bulan aktif:** {selected_month_str} ({len(df_filtered_month)} saham tersedia)")
    
    st.markdown("**Aliran Dana Kumulatif (Rp)**")
    fig_macro = px.line(df_cum_flow, x='Date', y='Cumulative Flow (Rp)', color='Kategori', title='Akumulasi Net Flow Lokal vs Asing')
    fig_macro.update_layout(
        hovermode="x unified", 
        yaxis_tickformat=',.0f',
        height=500
    )
    fig_macro.update_traces(hovertemplate='Tanggal: %{x|%b %Y}<br>Flow: Rp %{y:,.0f}')
    st.plotly_chart(fig_macro, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Net Buy (Kategori Investor)**")
        fig_buy = px.bar(df_net_flow.head(10), x='Total Net Flow (Rp)', y='Kategori', orientation='h',
                        color_discrete_sequence=['#00CC96'])
        fig_buy.update_layout(xaxis_tickformat=',.0f', height=400)
        st.plotly_chart(fig_buy, use_container_width=True)
    with c2:
        st.markdown("**Top Net Sell (Kategori Investor)**")
        fig_sell = px.bar(df_net_flow.tail(10), x='Total Net Flow (Rp)', y='Kategori', orientation='h',
                         color_discrete_sequence=['#EF553B'])
        fig_sell.update_layout(xaxis_tickformat=',.0f', height=400)
        st.plotly_chart(fig_sell, use_container_width=True)

# --- TAB 2: SEKTOR ---
with tab2:
    st.subheader(f"Rotasi Sektor - {selected_month_str}")
    
    # Filter untuk bulan tertentu
    df_month_tab2 = df_filtered_month.copy()
    
    cats = sorted([c.replace('_chg_Rp','') for c in OWNERSHIP_CHG_RP_COLS])
    sel_cat = st.selectbox("Pilih Kategori Investor:", cats, 
                          index=cats.index('Foreign IB') if 'Foreign IB' in cats else 0,
                          key="tab2_cat")
    
    df_sec_flow, msg = calculate_sector_rotation(df_month_tab2, sel_cat)
    
    if not df_sec_flow.empty:
        st.metric(f"Total Flow {sel_cat}", 
                 f"Rp {df_sec_flow['Net Flow (Rp)'].sum():,.0f}",
                 delta=f"{len(df_sec_flow)} sektor")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Top Buy Sektor oleh {sel_cat}**")
            fig_buy = px.bar(df_sec_flow.head(10), x='Net Flow (Rp)', y='Sector', orientation='h', 
                            color_discrete_sequence=['green'], text_auto=',.0f')
            fig_buy.update_layout(
                yaxis={'categoryorder':'total ascending'}, 
                xaxis_tickformat=',.0f',
                height=450
            )
            st.plotly_chart(fig_buy, use_container_width=True)
        with c2:
            st.markdown(f"**Top Sell Sektor oleh {sel_cat}**")
            fig_sell = px.bar(df_sec_flow.tail(10), x='Net Flow (Rp)', y='Sector', orientation='h', 
                             color_discrete_sequence=['red'], text_auto=',.0f')
            fig_sell.update_layout(
                yaxis={'categoryorder':'total descending'}, 
                xaxis_tickformat=',.0f',
                height=450
            )
            st.plotly_chart(fig_sell, use_container_width=True)
    else:
        st.warning(f"Tidak ada data untuk bulan {selected_month_str}")

# --- TAB 3: INDIVIDUAL ---
with tab3:
    st.subheader("Deep Dive Saham")
    
    # Filter saham yang ada di bulan terpilih
    stocks_in_month = sorted(df_filtered_month['Code'].unique())
    if len(stocks_in_month) == 0:
        stocks_in_month = sorted(df['Code'].unique())
    
    sel_stock = st.selectbox("Pilih Saham:", stocks_in_month, 
                           index=stocks_in_month.index('BBRI') if 'BBRI' in stocks_in_month else 0,
                           key="tab3_stock")
    
    if sel_stock:
        # Filter untuk saham tertentu
        df_stock_all = df[df['Code'] == sel_stock].sort_values('Date')
        df_stock_month = df_stock_all[df_stock_all['Date'].dt.strftime('%Y-%m') == selected_month_str]
        
        if not df_stock_month.empty:
            df_state, last_row = get_stock_ownership_state(df_stock_month, sel_stock)
        else:
            df_state, last_row = get_stock_ownership_state(df_stock_all, sel_stock)
            st.info(f"⚠️ Data untuk {sel_stock} tidak tersedia di {selected_month_str}. Menampilkan data terbaru.")
        
        # 1. Metrics (Top)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Harga Terakhir", f"Rp {last_row['Price']:,.0f}")
        with c2:
            st.metric("Free Float", f"{last_row['Free Float']}%")
        with c3:
            st.metric("Sektor", last_row['Sector'])
        with c4:
            month_flow = df_stock_month['Total_chg_Rp'].sum() if not df_stock_month.empty else 0
            st.metric(f"Flow {selected_month_str}", 
                     format_id_short(month_flow, is_currency=True),
                     delta_color="off")
        
        st.markdown("---")
        
        # 2. Distribution Layout
        c_sun, c_bar = st.columns([1.5, 1])
        with c_sun:
            st.markdown("**Peta Kepemilikan (Sunburst)**")
            sec_num = last_row.get('Sec. Num', 0)
            total_local = last_row.get('Total_Local', 0)
            total_foreign = last_row.get('Total_Foreign', 0)
            gap_shares = max(0, sec_num - total_local - total_foreign)
            
            labels = [f"Total\n({format_id_short(sec_num)})"]
            parents = [""]
            values = [sec_num]
            
            if total_local > 0:
                labels.append(f"Lokal\n({format_id_short(total_local)})")
                parents.append(labels[0])
                values.append(total_local)
            
            if total_foreign > 0:
                labels.append(f"Asing\n({format_id_short(total_foreign)})")
                parents.append(labels[0])
                values.append(total_foreign)
            
            if gap_shares > 0:
                labels.append(f"Warkat\n({format_id_short(gap_shares)})")
                parents.append(labels[0])
                values.append(gap_shares)
            
            for index, row in df_state.iterrows():
                if row['Jumlah Saham'] > 0:
                    cat_name = row['Kategori']
                    parent_node = f"Lokal\n({format_id_short(total_local)})" if "Local" in cat_name else f"Asing\n({format_id_short(total_foreign)})"
                    
                    if parent_node in labels:
                        disp_name = cat_name.replace('Local ','').replace('Foreign ','')
                        fmt_val = format_id_short(row['Jumlah Saham'])
                        child_label = f"{disp_name}\n({fmt_val})"
                        labels.append(child_label)
                        parents.append(parent_node)
                        values.append(row['Jumlah Saham'])
            
            fig_sun = go.Figure(go.Sunburst(
                labels=labels, parents=parents, values=values, 
                branchvalues="total",
                textinfo="label+percent parent",
                insidetextorientation='radial',
                hoverinfo="label+value+percent parent"
            ))
            fig_sun.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=450)
            st.plotly_chart(fig_sun, use_container_width=True)
            
        with c_bar:
            st.markdown("**Top 5 Holders (Volume)**")
            top_h = df_state.sort_values('Jumlah Saham', ascending=False).head(5).copy()
            if not top_h.empty:
                top_h['Label'] = top_h['Jumlah Saham'].apply(lambda x: format_id_short(x))
                fig_hbar = px.bar(top_h, x='Jumlah Saham', y='Kategori', orientation='h', 
                                 text='Label', color='Jumlah Saham')
                fig_hbar.update_layout(yaxis={'categoryorder':'total ascending'}, height=400,
                                      showlegend=False)
                st.plotly_chart(fig_hbar, use_container_width=True)
            else:
                st.info("Tidak ada data kepemilikan")
        
        st.markdown("---")
        
        # 3. SANKEY FLOW untuk bulan terpilih
        st.subheader(f"🔄 Visualisasi Aliran Dana - {selected_month_str}")
        
        if not df_stock_month.empty:
            col_f1, col_f2 = st.columns([1, 4])
            with col_f1:
                st.markdown("##### Filter Flow")
                mode_sankey = st.radio("Satuan:", ["Value (Rp)", "Volume (Lembar)"], horizontal=True)
                mode_key = 'Value' if 'Rp' in mode_sankey else 'Volume'
                
                # Tampilkan summary
                month_date = df_stock_month.iloc[0]['Date']
                total_flow = df_stock_month['Total_chg_Rp'].sum()
                st.metric("Total Flow Bulan Ini", 
                         format_id_short(total_flow, is_currency=True),
                         "NET BUY" if total_flow > 0 else "NET SELL")
            
            with col_f2:
                fig_sankey = create_sankey_chart(df_stock_month, sel_stock, month_date, mode=mode_key)
                if fig_sankey:
                    st.plotly_chart(fig_sankey, use_container_width=True)
                else:
                    st.info(f"Tidak ada perubahan kepemilikan signifikan pada bulan {selected_month_str}.")
        else:
            st.warning(f"Tidak ada data untuk {sel_stock} di bulan {selected_month_str}")
        
        st.markdown("---")
        
        # 4. Table Perubahan Bulanan
        st.markdown("### 📅 Detail Perubahan Bulanan (Volume Lembar)")
        df_m_chg = calculate_monthly_change_table(df_stock_all)
        
        # Highlight bulan yang sedang dipilih
        def highlight_selected_month(row):
            style = [''] * len(row)
            if row['Bulan'] == selected_month.strftime('%b %Y'):
                style[0] = 'background-color: #FFFACD; font-weight: bold;'
            return style
        
        styled_df = df_m_chg.style.apply(highlight_selected_month, axis=1)\
                                  .apply(highlight_max_min, subset=OWNERSHIP_CHG_COLS, axis=1)\
                                  .format("{:,.0f}", subset=OWNERSHIP_CHG_COLS)
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

# --- TAB 4: SCREENER ---
with tab4:
    st.subheader("Screener Big Rotation")
    
    # Filter untuk bulan tertentu
    df_month_tab4 = df_filtered_month.copy()
    
    if not df_month_tab4.empty:
        st.markdown(f"**Tren Flow Sektor - {selected_month_str}**")
        
        # Hitung flow per sektor untuk bulan ini
        sector_flow_month = df_month_tab4.groupby('Sector')['Total_chg_Rp'].sum().reset_index()
        sector_flow_month = sector_flow_month.sort_values('Total_chg_Rp', ascending=False)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Buying Sectors**")
            top_buy = sector_flow_month.head(10).copy()
            top_buy = top_buy[top_buy['Total_chg_Rp'] > 0]
            if not top_buy.empty:
                fig_buy_sec = px.bar(top_buy, x='Total_chg_Rp', y='Sector', orientation='h',
                                    color_discrete_sequence=['#2E8B57'])
                fig_buy_sec.update_layout(xaxis_tickformat=',.0f', height=400)
                st.plotly_chart(fig_buy_sec, use_container_width=True)
        
        with c2:
            st.markdown("**Top Selling Sectors**")
            top_sell = sector_flow_month.tail(10).copy()
            top_sell = top_sell[top_sell['Total_chg_Rp'] < 0]
            if not top_sell.empty:
                fig_sell_sec = px.bar(top_sell, x='Total_chg_Rp', y='Sector', orientation='h',
                                     color_discrete_sequence=['#DC143C'])
                fig_sell_sec.update_layout(xaxis_tickformat=',.0f', height=400)
                st.plotly_chart(fig_sell_sec, use_container_width=True)
    
    st.markdown(f"**Saham dengan Rotasi > Rp {min_rotation_value:,.0f} - {selected_month_str}**")
    
    # Filter Logic untuk bulan tertentu
    df_scr = df_month_tab4.copy()
    if not df_scr.empty:
        mask = (df_scr['Top_Buyer_Value_Rp'].abs() >= min_rotation_value) | (df_scr['Top_Seller_Value_Rp'].abs() >= min_rotation_value)
        df_scr = df_scr[mask].sort_values('Top_Buyer_Value_Rp', ascending=False)
        
        if not df_scr.empty:
            disp_cols = ['Code', 'Sector', 'Top_Buyer', 'Top_Buyer_Value_Rp', 'Top_Seller', 'Top_Seller_Value_Rp', 'Price', 'Total_chg_Rp']
            
            rename_map = {
                'Top_Buyer_Value_Rp': 'Value Buyer (Rp)',
                'Top_Seller_Value_Rp': 'Value Seller (Rp)',
                'Price': 'Harga (Rp)',
                'Total_chg_Rp': 'Net Flow (Rp)'
            }
            df_display_scr = df_scr[disp_cols].rename(columns=rename_map)
            cols_to_fmt = ['Value Buyer (Rp)', 'Value Seller (Rp)', 'Harga (Rp)', 'Net Flow (Rp)']
            
            # Format dan highlight
            def color_net_flow(val):
                if pd.isna(val): return ''
                if val > 0: return 'color: green; font-weight: bold;'
                elif val < 0: return 'color: red; font-weight: bold;'
                return ''
            
            styled_scr = df_display_scr.style.format("{:,.0f}", subset=cols_to_fmt)\
                                          .applymap(color_net_flow, subset=['Net Flow (Rp)'])
            
            st.dataframe(styled_scr, use_container_width=True, hide_index=True)
        else:
            st.info(f"Tidak ada saham dengan rotasi > Rp {min_rotation_value:,.0f} di bulan {selected_month_str}")
    else:
        st.warning(f"Tidak ada data untuk bulan {selected_month_str}")

# --- TAB 5: SINYAL POTENSIAL ---
with tab5:
    st.subheader("💎 Radar Saham Potensial (Smart Money Flow)")
    st.info("Mendeteksi akumulasi Smart Money (Asing+Institusi) vs Distribusi Ritel (Local ID).")
    
    # Filter data untuk beberapa bulan terakhir
    latest_date = df['Date'].max()
    date_cutoff = latest_date - pd.DateOffset(months=6)
    df_recent = df[df['Date'] >= date_cutoff].copy()
    
    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        lookback = st.slider("Window Analisis (Bulan)", 1, 6, 3, key="tab5_lookback")
        min_acc = st.number_input("Min. Akumulasi (Rp Miliar)", value=5.0, step=1.0, key="tab5_minacc") * 1_000_000_000
        st.caption(f"Filter: > Rp {format_id_short(min_acc, is_currency=True)}")
        
        # Tambahkan filter sektor
        all_sectors = ['All'] + sorted(df_recent['Sector'].unique().tolist())
        selected_sector = st.selectbox("Filter Sektor:", all_sectors, key="tab5_sector")
        
    df_sig = calculate_smart_money_signals(df_recent, window_periods=lookback, min_acc_threshold=min_acc)
    
    if not df_sig.empty:
        # Filter berdasarkan sektor
        if selected_sector != 'All':
            df_sig = df_sig[df_sig['Sector'] == selected_sector]
        
        df_accum = df_sig[df_sig['Smart Money Flow (Rp)'] >= min_acc].copy()
        
        with col_s2:
            st.metric("Saham Terdeteksi", f"{len(df_accum)} Emitter", 
                     f"Sektor: {selected_sector}")
            
        if not df_accum.empty:
            st.markdown("### 🏆 Top Picks: Big Accumulation")
            
            df_show = df_accum[['Code', 'Signal', 'Price', 'Price Chg (Window)%', 'Smart Money Flow (Rp)', 'Retail Flow (Rp)', 'Sector']].copy()
            df_show = df_show.rename(columns={
                'Price': 'Harga (Rp)',
                'Smart Money Flow (Rp)': 'Smart Money (Rp)',
                'Retail Flow (Rp)': 'Retail Flow (Rp)'
            })
            
            # Format dan highlight
            def color_signal(val):
                if '🔥' in str(val): return 'background-color: #FFD700; font-weight: bold;'
                elif '💎' in str(val): return 'background-color: #87CEEB; font-weight: bold;'
                elif '⚠️' in str(val): return 'background-color: #FFB6C1; font-weight: bold;'
                return ''
            
            fmt_cols = ['Harga (Rp)', 'Smart Money (Rp)', 'Retail Flow (Rp)']
            
            styled_sig = df_show.style.format("{:,.0f}", subset=fmt_cols)\
                                     .format("{:.2f}%", subset=['Price Chg (Window)%'])\
                                     .applymap(color_signal, subset=['Signal'])
            
            st.dataframe(styled_sig, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("### 🎯 Divergence Map (Flow vs Price)")
            
            fig_scat = px.scatter(
                df_accum, x="Smart Money Flow (Rp)", y="Price Chg (Window)%", 
                color="Signal", size="Price", hover_data=['Code', 'Sector'], text="Code",
                title=f"Flow vs Price Action ({lookback} Bulan Terakhir)",
                color_discrete_map={
                    '🔥 Big Accumulation': '#FF4500',
                    '💎 Divergence (Collect)': '#1E90FF',
                    '⚠️ Distribution': '#DC143C'
                }
            )
            fig_scat.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_scat.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_scat.update_traces(textposition='top center', marker=dict(opacity=0.8))
            fig_scat.update_layout(xaxis_tickformat=',.0f', height=500)
            fig_scat.update_traces(hovertemplate='<b>%{text}</b><br>Sektor: %{customdata[1]}<br>Flow: Rp %{x:,.0f}<br>Chg: %{y:.2f}%<br>Signal: %{marker.color}')
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.warning("Tidak ada sinyal yang memenuhi kriteria dengan filter saat ini.")
    else:
        st.warning("Belum ada sinyal yang memenuhi kriteria.")

# --- TAB 6: 🔥 NEW - PERGERAKAN SIGNIFIKAN PER BULAN ---
with tab6:
    st.subheader(f"🔥 Pergerakan Signifikan - {selected_month_str}")
    st.info("Mendeteksi saham dengan pergerakan kepemilikan signifikan di bulan tertentu.")
    
    # Tampilkan info bulan
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Bulan", selected_month_str)
    with col_info2:
        st.metric("Total Saham", f"{len(df_filtered_month):,}")
    with col_info3:
        unique_sectors = df_filtered_month['Sector'].nunique()
        st.metric("Sektor", unique_sectors)
    
    # Hitung significant movements
    if not df_filtered_month.empty:
        with st.spinner(f"Menganalisis {len(df_filtered_month)} saham..."):
            df_significant = get_significant_movements(
                df_filtered_month, 
                threshold_rp=threshold_rp,
                threshold_pct=threshold_pct
            )
        
        if not df_significant.empty:
            st.success(f"✅ Ditemukan {len(df_significant)} saham dengan pergerakan signifikan")
            
            # Summary metrics
            total_flow = df_significant['Total Flow (Rp)'].sum()
            net_buy = df_significant[df_significant['Net Flow (Rp)'] > 0]['Net Flow (Rp)'].sum()
            net_sell = df_significant[df_significant['Net Flow (Rp)'] < 0]['Net Flow (Rp)'].sum()
            
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            with col_sum1:
                st.metric("Total Flow", format_id_short(total_flow, is_currency=True))
            with col_sum2:
                st.metric("Net Buy", format_id_short(net_buy, is_currency=True))
            with col_sum3:
                st.metric("Net Sell", format_id_short(abs(net_sell), is_currency=True))
            with col_sum4:
                buy_count = len(df_significant[df_significant['Direction'] == 'NET BUY'])
                sell_count = len(df_significant[df_significant['Direction'] == 'NET SELL'])
                st.metric("Buy/Sell Ratio", f"{buy_count}:{sell_count}")
            
            st.markdown("---")
            
            # Tabs untuk tampilan berbeda
            tab6_1, tab6_2, tab6_3 = st.tabs(["📊 Tabel Detail", "📈 Visualisasi", "🎯 Top Movers"])
            
            with tab6_1:
                # Tabel detail
                st.markdown("### Detail Pergerakan Signifikan")
                
                display_cols = ['Code', 'Sector', 'Price', 'Total Flow (Rp)', 'Net Flow (Rp)', 
                              'Flow % of Shares', 'Direction', 'Top_Buyer', 'Top_Seller']
                
                df_display = df_significant[display_cols].copy()
                
                # Formatting
                def color_direction(val):
                    if val == 'NET BUY': return 'color: green; font-weight: bold;'
                    elif val == 'NET SELL': return 'color: red; font-weight: bold;'
                    return 'color: gray;'
                
                styled_table = df_display.style.format({
                    'Price': '{:,.0f}',
                    'Total Flow (Rp)': '{:,.0f}',
                    'Net Flow (Rp)': '{:,.0f}',
                    'Flow % of Shares': '{:.2f}%'
                }).applymap(color_direction, subset=['Direction'])
                
                st.dataframe(styled_table, use_container_width=True, hide_index=True)
            
            with tab6_2:
                # Visualisasi
                st.markdown("### Heatmap Sektor")
                
                # Group by sector
                sector_flow = df_significant.groupby('Sector').agg({
                    'Total Flow (Rp)': 'sum',
                    'Net Flow (Rp)': 'sum',
                    'Code': 'count'
                }).rename(columns={'Code': 'Jumlah Saham'}).reset_index()
                
                fig_heat = px.treemap(sector_flow, path=['Sector'], values='Total Flow (Rp)',
                                     color='Net Flow (Rp)', color_continuous_scale='RdBu',
                                     hover_data=['Jumlah Saham', 'Net Flow (Rp)'],
                                     title=f"Distribusi Flow per Sektor - {selected_month_str}")
                fig_heat.update_traces(textinfo="label+value+percent parent")
                fig_heat.update_layout(height=600)
                st.plotly_chart(fig_heat, use_container_width=True)
                
                # Scatter plot: Flow vs % Shares
                st.markdown("### Flow vs % Saham")
                fig_scatter = px.scatter(df_significant, x='Flow % of Shares', y='Total Flow (Rp)',
                                        color='Direction', size='Price', hover_data=['Code', 'Sector'],
                                        color_discrete_map={'NET BUY': 'green', 'NET SELL': 'red', 'NEUTRAL': 'gray'},
                                        title="Intensitas Pergerakan")
                fig_scatter.update_layout(xaxis_tickformat='.2f', yaxis_tickformat=',.0f', height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab6_3:
                # Top Movers
                st.markdown("#### 🥇 Top 10 Buyers")
                top_buyers = df_significant[df_significant['Direction'] == 'NET BUY']\
                    .nlargest(10, 'Total Flow (Rp)')
                
                if not top_buyers.empty:
                    fig_top_buy = px.bar(top_buyers, x='Code', y='Total Flow (Rp)', 
                                        color='Flow % of Shares',
                                        hover_data=['Sector', 'Top_Buyer'],
                                        color_continuous_scale='greens')
                    fig_top_buy.update_layout(xaxis_tickformat=',.0f', height=400)
                    st.plotly_chart(fig_top_buy, use_container_width=True)
                
                st.markdown("#### 🥇 Top 10 Sellers")
                top_sellers = df_significant[df_significant['Direction'] == 'NET SELL']\
                    .nlargest(10, 'Total Flow (Rp)')
                
                if not top_sellers.empty:
                    fig_top_sell = px.bar(top_sellers, x='Code', y='Total Flow (Rp)', 
                                         color='Flow % of Shares',
                                         hover_data=['Sector', 'Top_Seller'],
                                         color_continuous_scale='reds')
                    fig_top_sell.update_layout(xaxis_tickformat=',.0f', height=400)
                    st.plotly_chart(fig_top_sell, use_container_width=True)
            
            # Download button untuk data significant movements
            st.markdown("---")
            st.markdown("### 📥 Export Data")
            csv_significant = df_significant.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data Pergerakan Signifikan",
                data=csv_significant,
                file_name=f"significant_movements_{selected_month_str}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.warning(f"Tidak ditemukan saham dengan pergerakan signifikan di bulan {selected_month_str}")
            st.info(f"Coba turunkan threshold (saat ini: Rp {threshold_rp:,.0f} atau {threshold_pct}% dari saham)")
    else:
        st.warning(f"Tidak ada data untuk bulan {selected_month_str}")

# ==============================================================================
# 🎯 FOOTER
# ==============================================================================
st.divider()
st.caption("📊 Dashboard Bandarmology KSEI v2.0 • Data: KSEI Shareholder Processed • Updated with Month Filter")
