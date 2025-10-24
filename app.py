# ==============================================================================
# 📦 1) IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io

# Import library Google
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==============================================================================
# ⚙️ 2) KONFIGURASI DASHBOARD & G-DRIVE
# ==============================================================================
st.set_page_config(
    page_title="🌊 Dashboard Analisis Aliran Dana KSEI",
    layout="wide",
    page_icon="🌊"
)

# --- KONFIGURASI G-DRIVE ---
# Kita pakai folder yang sama dengan proyek sebelumnya
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP" 
# TAPI, kita cari file KSEI yang baru
FILE_NAME = "KSEI_Shareholder_Processed.csv" 

# --- KONFIGURASI KATEGORI (PENTING) ---
# 18 Kategori Kepemilikan (untuk Tab 1 dan 2)
OWNERSHIP_COLS = [
    'Local IS', 'Local CP', 'Local PF', 'Local IB', 'Local ID', 'Local MF', 'Local SC', 'Local FD', 'Local OT',
    'Foreign IS', 'Foreign CP', 'Foreign PF', 'Foreign IB', 'Foreign ID', 'Foreign MF', 'Foreign SC', 'Foreign FD', 'Foreign OT'
]
# Kolom perubahan (untuk Tab 1)
OWNERSHIP_CHG_COLS = [f"{col}_chg" for col in OWNERSHIP_COLS]

# ==============================================================================
# 📦 3) FUNGSI MEMUAT DATA (via SERVICE ACCOUNT)
# ==============================================================================
# Kita bisa REUSE fungsi otentikasi dari proyek sebelumnya
def get_gdrive_service():
    try:
        creds_json = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_json, scopes=['https://www.googleapis.com/auth/drive.readonly'])
        service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        return service, None
    except KeyError:
        msg = "❌ Gagal otentikasi: 'st.secrets' tidak menemukan key [gcp_service_account]. Pastikan 'secrets.toml' sudah benar."
        return None, msg
    except Exception as e:
        msg = f"❌ Gagal otentikasi Google Drive: {e}."
        return None, msg

@st.cache_data(ttl=3600) # Cache data selama 1 jam
def load_data():
    """Mencari file KSEI, men-download, membersihkan, dan membacanya ke Pandas."""
    service, error_msg = get_gdrive_service()
    if error_msg:
        return pd.DataFrame(), error_msg, "error"

    try:
        # 1. Cari file ID terbaru
        query = f"'{FOLDER_ID}' in parents and name='{FILE_NAME}' and trashed=false"
        results = service.files().list(
            q=query, fields="files(id, name)", orderBy="modifiedTime desc", pageSize=1
        ).execute()
        items = results.get('files', [])

        if not items:
            msg = f"❌ File '{FILE_NAME}' tidak ditemukan di folder GDrive. Pastikan skrip Colab sudah berjalan dan 'Share' folder sudah benar."
            return pd.DataFrame(), msg, "error"

        file_id = items[0]['id']
        
        # 2. Download file
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)

        # 3. Baca ke Pandas
        df = pd.read_csv(fh, dtype=object) # Baca semua sebagai string dulu
        
        # 4. Pembersihan Data (Sangat Penting)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Tentukan semua kolom yang seharusnya ANGKA
        cols_to_numeric = [
            'Price', 'Price_Chg %', 'Free Float', 'Total_Local', 'Total_Foreign',
            'Top_Buyer_Vol', 'Top_Seller_Vol'
        ] + OWNERSHIP_COLS + OWNERSHIP_CHG_COLS # Tambahkan semua kolom _chg dan raw

        for col in cols_to_numeric:
            if col in df.columns:
                # Bersihkan string "kotor" (koma, dll)
                cleaned_col = df[col].astype(str).str.strip()
                cleaned_col = cleaned_col.str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0)

        df = df.dropna(subset=['Date', 'Code'])
        
        # Buat kolom helper (Total Local/Foreign Change)
        local_chg_cols = [col for col in OWNERSHIP_CHG_COLS if 'Local' in col]
        foreign_chg_cols = [col for col in OWNERSHIP_CHG_COLS if 'Foreign' in col]
        
        df['Total_Local_chg'] = df[local_chg_cols].sum(axis=1)
        df['Total_Foreign_chg'] = df[foreign_chg_cols].sum(axis=1)

        msg = f"Data KSEI berhasil dimuat (file ID: {file_id})."
        return df, msg, "success"
    
    except Exception as e:
        msg = f"❌ Terjadi error saat memuat data KSEI: {e}."
        return pd.DataFrame(), msg, "error"

# ==============================================================================
# 🛠️ 4) FUNGSI KALKULASI (untuk Tabs)
# ==============================================================================

@st.cache_data
def calculate_macro_flow(df, start_date, end_date):
    """(TAB 1) Menghitung total aliran dana per kategori di seluruh market."""
    df_period = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # 1. Total Net Flow per Kategori
    net_flow = df_period[OWNERSHIP_CHG_COLS].sum().reset_index()
    net_flow.columns = ['Kategori', 'Total Net Flow (Shares)']
    net_flow['Kategori'] = net_flow['Kategori'].str.replace('_chg', '')
    net_flow = net_flow.sort_values(by='Total Net Flow (Shares)', ascending=False)
    
    # 2. Cumulative Flow (Local vs Foreign)
    cum_flow = df_period.groupby('Date')[['Total_Local_chg', 'Total_Foreign_chg']].sum().cumsum().reset_index()
    cum_flow = cum_flow.melt('Date', var_name='Kategori', value_name='Cumulative Flow')
    cum_flow['Kategori'] = cum_flow['Kategori'].str.replace('_chg', '')
    
    return net_flow, cum_flow

@st.cache_data
def get_stock_ownership_state(df, stock_code):
    """(TAB 2) Mengambil data kepemilikan TERBARU untuk 1 saham."""
    df_stock = df[df['Code'] == stock_code]
    if df_stock.empty:
        return pd.DataFrame()
        
    # Ambil data TANGGAL TERBARU
    latest_state = df_stock.sort_values('Date').iloc[-1]
    
    # Ubah dari format 'wide' (banyak kolom) ke 'long' (2 kolom)
    df_state = latest_state[OWNERSHIP_COLS].reset_index()
    df_state.columns = ['Kategori', 'Jumlah Saham']
    
    # Pisahkan Local/Foreign untuk grouping
    df_state['Tipe'] = df_state['Kategori'].apply(lambda x: 'Local' if 'Local' in x else 'Foreign')
    df_state['Kategori'] = df_state['Kategori'].str.replace('Local ', '').str.replace('Foreign ', '')
    
    # Hitung total untuk persentase
    total_shares = df_state['Jumlah Saham'].sum()
    if total_shares > 0:
        df_state['Persentase'] = (df_state['Jumlah Saham'] / total_shares) * 100
    else:
        df_state['Persentase'] = 0
        
    return df_state.sort_values(by='Jumlah Saham', ascending=False)


# ==============================================================================
# 💎 5) LAYOUT UTAMA (HEADER)
# ==============================================================================
st.title("🌊 Dashboard Analisis Aliran Dana KSEI")
st.caption("Menganalisis rotasi kepemilikan saham (flow) untuk mengambil keputusan.")

# Panggil data dan tangkap statusnya
df, status_msg, status_level = load_data()

# Tampilkan notifikasi
if status_level == "success":
    st.toast(status_msg, icon="✅")
elif status_level == "error":
    st.error(status_msg)

# ==============================================================================
# 🧭 6) SIDEBAR FILTER
# ==============================================================================
st.sidebar.header("🎛️ Filter Analisis")

if st.sidebar.button("🔄 Refresh Data (Tarik Ulang dari GDrive)"):
    st.cache_data.clear()
    st.rerun()

if df.empty:
    st.warning("⚠️ Data KSEI belum berhasil dimuat. Dashboard tidak dapat dilanjutkan.")
    st.stop()

# --- Filter Utama ---
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

selected_date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(max_date - pd.Timedelta(days=365), max_date), # Default 1 tahun
    min_value=min_date,
    max_value=max_date,
    format="DD-MM-YYYY"
)

# Konversi ke datetime untuk perbandingan
try:
    start_date = pd.to_datetime(selected_date_range[0])
    end_date = pd.to_datetime(selected_date_range[1])
except:
    st.sidebar.error("Rentang tanggal tidak valid. Menggunakan data 1 tahun terakhir.")
    start_date = pd.to_datetime(max_date - pd.Timedelta(days=365))
    end_date = pd.to_datetime(max_date)

# --- Filter untuk Tab 3 (Screener) ---
st.sidebar.header("Filter Screener (u/ Tab 3)")

all_stocks = sorted(df['Code'].unique())
selected_stocks = st.sidebar.multiselect(
    "Filter Saham:",
    all_stocks,
    placeholder="Ketik kode saham"
)

all_categories = sorted(OWNERSHIP_COLS)
selected_buyers = st.sidebar.multiselect(
    "Filter Top Buyer:",
    all_categories,
    placeholder="Cari pergerakan oleh..."
)

selected_sellers = st.sidebar.multiselect(
    "Filter Top Seller:",
    all_categories,
    placeholder="Cari pergerakan oleh..."
)

min_rotation_vol = st.sidebar.number_input(
    "Minimum Volume Rotasi (Saham)",
    min_value=0,
    value=1000000, # Default 1 juta lembar
    step=100000
)

# --- Terapkan Filter ---
df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

if selected_stocks:
    df_filtered = df_filtered[df_filtered['Code'].isin(selected_stocks)]
if selected_buyers:
    df_filtered = df_filtered[df_filtered['Top_Buyer'].isin(selected_buyers)]
if selected_sellers:
    df_filtered = df_filtered[df_filtered['Top_Seller'].isin(selected_sellers)]
if min_rotation_vol > 0:
    # Filter jika Top_Buyer_Vol > min ATAU Top_Seller_Vol (abs) > min
    df_filtered = df_filtered[
        (df_filtered['Top_Buyer_Vol'] >= min_rotation_vol) |
        (df_filtered['Top_Seller_Vol'].abs() >= min_rotation_vol)
    ]

# ==============================================================================
#  LAYOUT UTAMA (DENGAN TABS)
# ==============================================================================
st.caption(f"Menampilkan data dari **{start_date.strftime('%d %b %Y')}** hingga **{end_date.strftime('%d %b %Y')}**")

tab1, tab2, tab3 = st.tabs([
    "🌊 **Ringkasan Aliran Dana (Makro)**",
    "📈 **Analisis Saham Individual (Mikro)**",
    "🔍 **Screener Rotasi Kepemilikan**" 
])

# --- TAB 1: RINGKASAN ALIRAN DANA (MAKRO) ---
with tab1:
    st.subheader("Bagaimana Peta Aliran Dana di Seluruh Market?")
    st.info(f"Menganalisis total pergerakan bersih (net flow) untuk setiap kategori investor di seluruh saham dalam rentang tanggal terpilih.")
    
    # Hitung data makro
    df_net_flow, df_cum_flow = calculate_macro_flow(df, start_date, end_date)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Total Net Flow per Kategori**")
        
        # Format manual untuk tabel
        df_net_flow_display = df_net_flow.copy()
        df_net_flow_display['Total Net Flow (Shares)'] = df_net_flow_display['Total Net Flow (Shares)'].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A'
        )
        
        st.dataframe(
            df_net_flow_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Kategori": st.column_config.TextColumn("Kategori Investor"),
                "Total Net Flow (Shares)": st.column_config.TextColumn("Total Net Flow")
            }
        )
    
    with col2:
        st.markdown("**Aliran Dana Kumulatif (Lokal vs Asing)**")
        fig_macro = px.line(
            df_cum_flow,
            x='Date',
            y='Cumulative Flow',
            color='Kategori',
            title='Aliran Kumulatif Lokal vs Asing (Total Market)',
            labels={'Cumulative Flow': 'Total Saham (Kumulatif)', 'Date': 'Tanggal'}
        )
        fig_macro.update_traces(hovertemplate='Tanggal: %{x}<br>Flow: %{y:,.0f}<extra></extra>')
        fig_macro.update_layout(hovermode="x unified")
        st.plotly_chart(fig_macro, use_container_width=True)

# --- TAB 2: ANALISIS SAHAM INDIVIDUAL (MIKRO) ---
with tab2:
    st.subheader("Bagaimana Aliran Dana di Satu Saham?")
    
    stock_to_analyze = st.selectbox(
        "Pilih Saham untuk Analisis Mendalam:",
        all_stocks,
        index=all_stocks.index("BBCA") if "BBCA" in all_stocks else 0
    )
    
    if stock_to_analyze:
        df_stock_filtered = df_filtered[df_filtered['Code'] == stock_to_analyze].sort_values('Date')
        df_state = get_stock_ownership_state(df, stock_to_analyze)
        
        if df_stock_filtered.empty or df_state.empty:
            st.warning(f"Tidak ada data untuk {stock_to_analyze} pada rentang tanggal terpilih.")
        else:
            latest_price = df_stock_filtered.iloc[-1]['Price']
            free_float = df_stock_filtered.iloc[-1]['Free Float']
            
            st.markdown(f"**Analisis: {stock_to_analyze}**")
            col1, col2 = st.columns(2)
            col1.metric("Harga Terakhir", f"Rp {latest_price:,.0f}" if pd.notna(latest_price) else "N/A")
            col2.metric("Free Float Saham", f"{free_float:.2f}%" if pd.notna(free_float) else "N/A")

            st.markdown("---")
            
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown("**Peta Kepemilikan (Terbaru)**")
                fig_pie = px.treemap(
                    df_state,
                    path=[px.Constant("Semua"), 'Tipe', 'Kategori'],
                    values='Jumlah Saham',
                    title=f'Komposisi Pemegang Saham {stock_to_analyze} (Terbaru)',
                    hover_data={'Persentase': ':.2f%'}
                )
                fig_pie.update_traces(textinfo="label+percent root")
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                st.markdown("**Rotasi Kepemilikan vs Harga**")
                
                # Buat subplot (Flow vs Price)
                fig_flow = make_subplots(specs=[[{"secondary_y": True}]])

                # Plot 1: Flow (Bar)
                fig_flow.add_trace(go.Bar(
                    x=df_stock_filtered['Date'],
                    y=df_stock_filtered['Top_Buyer_Vol'],
                    name='Top Buyer (Vol)',
                    marker_color='green',
                    customdata=df_stock_filtered['Top_Buyer'],
                    hovertemplate='Tanggal: %{x}<br>Buyer: %{customdata}<br>Vol: %{y:,.0f}<extra></extra>'
                ), secondary_y=False)
                
                fig_flow.add_trace(go.Bar(
                    x=df_stock_filtered['Date'],
                    y=df_stock_filtered['Top_Seller_Vol'],
                    name='Top Seller (Vol)',
                    marker_color='red',
                    customdata=df_stock_filtered['Top_Seller'],
                    hovertemplate='Tanggal: %{x}<br>Seller: %{customdata}<br>Vol: %{y:,.0f}<extra></extra>'
                ), secondary_y=False)

                # Plot 2: Harga (Line)
                fig_flow.add_trace(go.Scatter(
                    x=df_stock_filtered['Date'],
                    y=df_stock_filtered['Price'],
                    name='Harga Saham (Rp)',
                    line=dict(color='blue'),
                    hovertemplate='Tanggal: %{x}<br>Harga: %{y:,.0f}<extra></extra>'
                ), secondary_y=True)

                fig_flow.update_layout(
                    title=f'Aliran Dana (Buyer/Seller) vs Pergerakan Harga {stock_to_analyze}',
                    barmode='relative',
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig_flow.update_yaxes(title_text="Volume Rotasi (Shares)", secondary_y=False)
                fig_flow.update_yaxes(title_text="Harga Saham (Rp)", secondary_y=True)
                st.plotly_chart(fig_flow, use_container_width=True)

# --- TAB 3: SCREENER ROTASI ---
with tab3:
    st.subheader("Screener Rotasi Kepemilikan")
    st.info("Gunakan filter di sidebar untuk mencari saham dengan rotasi kepemilikan terbesar atau oleh kategori investor tertentu.")
    
    cols_to_display = [
        'Date', 'Code', 'Top_Buyer', 'Top_Buyer_Vol', 'Top_Seller', 'Top_Seller_Vol',
        'Price', 'Price_Chg %', 'Free Float'
    ]
    df_screener = df_filtered[cols_to_display].sort_values(by='Top_Buyer_Vol', ascending=False)
    
    # --- Format Manual untuk Tampilan ---
    df_screener_display = df_screener.copy()
    
    # Format string manual
    df_screener_display['Top_Buyer_Vol'] = df_screener_display['Top_Buyer_Vol'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
    df_screener_display['Top_Seller_Vol'] = df_screener_display['Top_Seller_Vol'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
    df_screener_display['Price'] = df_screener_display['Price'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
    
    st.dataframe(
        df_screener_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Tanggal", format="DD-MM-YYYY"),
            "Code": st.column_config.TextColumn("Saham"),
            "Top_Buyer": st.column_config.TextColumn("Top Buyer"),
            "Top_Buyer_Vol": st.column_config.TextColumn("Vol Buyer"),
            "Top_Seller": st.column_config.TextColumn("Top Seller"),
            "Top_Seller_Vol": st.column_config.TextColumn("Vol Seller"),
            "Price": st.column_config.TextColumn("Harga"),
            "Price_Chg %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
            "Free Float": st.column_config.NumberColumn("Free Float %", format="%.2f%%")
        }
    )
