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
FOLDER_ID = "1hX2jwUrAgi4Fr8xkcFWjCW6vbk6lsIlP" 
FILE_NAME = "KSEI_Shareholder_Processed.csv" 

# --- KONFIGURASI KATEGORI (PENTING) ---
OWNERSHIP_COLS = [
    'Local IS', 'Local CP', 'Local PF', 'Local IB', 'Local ID', 'Local MF', 'Local SC', 'Local FD', 'Local OT',
    'Foreign IS', 'Foreign CP', 'Foreign PF', 'Foreign IB', 'Foreign ID', 'Foreign MF', 'Foreign SC', 'Foreign FD', 'Foreign OT'
]
OWNERSHIP_CHG_COLS = [f"{col}_chg" for col in OWNERSHIP_COLS]

# ==============================================================================
# 📦 3) FUNGSI MEMUAT DATA (via SERVICE ACCOUNT)
# ==============================================================================
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

@st.cache_data(ttl=3600)
def load_data(): 
    """Mencari file KSEI, men-download, membersihkan, dan membacanya ke Pandas."""
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
        
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # [PERUBAHAN] Pastikan 'Sector' juga di-strip whitespace jika ada
        if 'Sector' in df.columns:
            df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
            df['Sector'] = 'Others' # Fallback jika kolom Sector entah kenapa tidak ada

        cols_to_numeric = [
            'Price', 'Price_Chg %', 'Free Float', 'Total_Local', 'Total_Foreign',
            'Top_Buyer_Vol', 'Top_Seller_Vol'
        ] + OWNERSHIP_COLS + OWNERSHIP_CHG_COLS

        for col in cols_to_numeric:
            if col in df.columns:
                cleaned_col = df[col].astype(str).str.strip()
                cleaned_col = cleaned_col.str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0)

        df = df.dropna(subset=['Date', 'Code'])
        
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
def calculate_macro_flow(df_filtered_by_year):
    """(TAB 1) Menghitung total aliran dana per kategori di seluruh market."""
    # 1. Total Net Flow per Kategori (untuk Bar Chart)
    net_flow = df_filtered_by_year[OWNERSHIP_CHG_COLS].sum().reset_index()
    net_flow.columns = ['Kategori', 'Total Net Flow (Shares)']
    net_flow['Kategori'] = net_flow['Kategori'].str.replace('_chg', '')
    net_flow = net_flow.sort_values(by='Total Net Flow (Shares)', ascending=False)
    
    # 2. Cumulative Flow (Local vs Foreign) (untuk Line Chart)
    cum_flow = df_filtered_by_year.groupby('Date')[['Total_Local_chg', 'Total_Foreign_chg']].sum().cumsum().reset_index()
    cum_flow = cum_flow.melt('Date', var_name='Kategori', value_name='Cumulative Flow')
    cum_flow['Kategori'] = cum_flow['Kategori'].str.replace('_chg', ' (Net)')
    
    return net_flow, cum_flow

# [BARU] Fungsi untuk analisis rotasi sektor
@st.cache_data
def calculate_sector_rotation(df_filtered_by_year, selected_category):
    """(TAB 2 BARU) Menghitung aliran dana bersih kategori tertentu per sektor."""
    
    if 'Sector' not in df_filtered_by_year.columns or df_filtered_by_year['Sector'].nunique() <= 1:
        return pd.DataFrame(), "Data sektor tidak tersedia atau hanya 'Others'. Pastikan kolom 'Sector' ada di file olahan KSEI."

    # Kolom perubahan untuk kategori yang dipilih
    category_chg_col = f"{selected_category}_chg"
    
    if category_chg_col not in df_filtered_by_year.columns:
        return pd.DataFrame(), f"Kolom '{category_chg_col}' tidak ditemukan dalam data."

    # Group by Sector dan hitung sum dari perubahan kategori yang dipilih
    sector_category_flow = df_filtered_by_year.groupby('Sector')[category_chg_col].sum().reset_index()
    sector_category_flow.columns = ['Sector', 'Net Flow (Shares)']
    sector_category_flow = sector_category_flow.sort_values(by='Net Flow (Shares)', ascending=False)
    
    return sector_category_flow, None

@st.cache_data
def get_stock_ownership_state(df, stock_code):
    """(TAB 3 - dulu Tab 2) Mengambil data kepemilikan TERBARU untuk 1 saham."""
    df_stock = df[df['Code'] == stock_code]
    if df_stock.empty:
        return pd.DataFrame()
        
    latest_state = df_stock.sort_values('Date').iloc[-1]
    df_state = latest_state[OWNERSHIP_COLS].reset_index()
    df_state.columns = ['Kategori', 'Jumlah Saham']
    
    df_state['Tipe'] = df_state['Kategori'].apply(lambda x: 'Local' if 'Local' in x else 'Foreign')
    df_state['Kategori'] = df_state['Kategori'].str.replace('Local ', '').str.replace('Foreign ', '')
    
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

# Panggil data
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
    load_data.clear() 
    st.rerun()

if df.empty:
    st.warning("⚠️ Data KSEI belum berhasil dimuat. Dashboard tidak dapat dilanjutkan.")
    st.stop()

# Filter Utama: TAHUN
all_years = sorted(df['Date'].dt.year.unique(), reverse=True)
max_year = df['Date'].dt.year.max()

selected_years = st.sidebar.multiselect(
    "Pilih Tahun Analisis", 
    options=all_years,
    default=[max_year] 
)

if not selected_years:
    st.sidebar.warning("Pilih minimal satu tahun.")
    selected_years = [max_year]

df_filtered_by_year = df[df['Date'].dt.year.isin(selected_years)].copy()
st.caption(f"Menampilkan data untuk tahun: **{', '.join(map(str, selected_years))}**")

# Filter untuk Tab 4 (Screener)
st.sidebar.header("Filter Screener (u/ Tab 3)") # Update nomor tab

all_stocks = sorted(df_filtered_by_year['Code'].unique())
selected_stocks = st.sidebar.multiselect(
    "Filter Saham:",
    all_stocks,
    placeholder="Ketik kode saham"
)

all_categories_base = [col.replace('_chg', '') for col in OWNERSHIP_CHG_COLS]
selected_buyers = st.sidebar.multiselect(
    "Filter Top Buyer:",
    sorted(all_categories_base),
    placeholder="Cari pergerakan oleh..."
)

selected_sellers = st.sidebar.multiselect(
    "Filter Top Seller:",
    sorted(all_categories_base),
    placeholder="Cari pergerakan oleh..."
)

min_rotation_vol = st.sidebar.number_input(
    "Minimum Volume Rotasi (Saham)",
    min_value=0,
    value=1000000, 
    step=100000
)

# Terapkan Filter (hanya untuk screener)
df_screener_filtered = df_filtered_by_year.copy()

if selected_stocks:
    df_screener_filtered = df_screener_filtered[df_screener_filtered['Code'].isin(selected_stocks)]
if selected_buyers:
    # Perlu memastikan kolom Top_Buyer berisi string Kategori tanpa _chg
    df_screener_filtered = df_screener_filtered[df_screener_filtered['Top_Buyer'].isin(selected_buyers)]
if selected_sellers:
    # Perlu memastikan kolom Top_Seller berisi string Kategori tanpa _chg
    df_screener_filtered = df_screener_filtered[df_screener_filtered['Top_Seller'].isin(selected_sellers)]
if min_rotation_vol > 0:
    df_screener_filtered = df_screener_filtered[
        (df_screener_filtered['Top_Buyer_Vol'] >= min_rotation_vol) |
        (df_screener_filtered['Top_Seller_Vol'].abs() >= min_rotation_vol)
    ]

# ==============================================================================
#  LAYOUT UTAMA (DENGAN 4 TABS BARU)
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🌊 **Makro (Market)**",
    "📊 **Analisis Sektor (Rotasi)**", # Tab Baru
    "📈 **Mikro (Saham)**",
    "🔍 **Screener Rotasi**" 
])

# --- TAB 1: RINGKASAN ALIRAN DANA (MARKET) ---
with tab1:
    st.subheader(f"Peta Aliran Dana Market (Tahun: {', '.join(map(str, selected_years))})")
    
    df_net_flow, df_cum_flow = calculate_macro_flow(df_filtered_by_year)
    
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
    
    st.markdown("---")
    st.markdown("**Kategori Investor Terkuat (Net Flow)**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 5 Kategori Net Buy**")
        top_5_buy = df_net_flow.head(5)
        fig_buy = px.bar(
            top_5_buy,
            x='Total Net Flow (Shares)',
            y='Kategori',
            orientation='h',
            text='Total Net Flow (Shares)'
        )
        fig_buy.update_layout(yaxis={'categoryorder':'total ascending'})
        fig_buy.update_traces(texttemplate='%{x:,.0f}', textposition='outside', marker_color='green')
        st.plotly_chart(fig_buy, use_container_width=True)

    with col2:
        st.markdown("**Top 5 Kategori Net Sell**")
        top_5_sell = df_net_flow.tail(5).sort_values(by='Total Net Flow (Shares)')
        fig_sell = px.bar(
            top_5_sell,
            x='Total Net Flow (Shares)',
            y='Kategori',
            orientation='h',
            text='Total Net Flow (Shares)'
        )
        fig_sell.update_layout(yaxis={'categoryorder':'total descending'})
        fig_sell.update_traces(texttemplate='%{x:,.0f}', textposition='outside', marker_color='red')
        st.plotly_chart(fig_sell, use_container_width=True)

# --- TAB 2: [BARU] ANALISIS SEKTOR (ROTASI) ---
with tab2:
    st.subheader(f"Analisis Rotasi Kategori Investor per Sektor (Tahun: {', '.join(map(str, selected_years))})")
    
    if 'Sector' not in df_filtered_by_year.columns or df_filtered_by_year['Sector'].nunique() <= 1:
        st.warning("Kolom 'Sector' tidak ditemukan atau hanya berisi 'Others'. Pastikan file olahan KSEI memiliki data sektor yang benar.")
    else:
        # Pilihan Kategori Investor untuk analisis sektor
        all_categories_for_sector = sorted([col.replace('_chg', '') for col in OWNERSHIP_CHG_COLS])
        selected_category_for_sector = st.selectbox(
            "Pilih Kategori Investor:",
            all_categories_for_sector,
            key="sector_category_select"
        )

        if selected_category_for_sector:
            df_sector_cat_flow, error_sec_cat = calculate_sector_rotation(df_filtered_by_year, selected_category_for_sector)
            
            if error_sec_cat:
                st.error(error_sec_cat)
            elif not df_sector_cat_flow.empty:
                st.markdown(f"**Net Flow ({selected_category_for_sector}) per Sektor**")
                
                # Buat dua kolom untuk Top Buy dan Top Sell Sektor
                col_sec_1, col_sec_2 = st.columns(2)
                
                with col_sec_1:
                    st.markdown(f"**Top 10 Sektor Net Buy oleh {selected_category_for_sector}**")
                    top_buy_sectors = df_sector_cat_flow[df_sector_cat_flow['Net Flow (Shares)'] > 0].head(10)
                    if not top_buy_sectors.empty:
                        fig_sec_buy = px.bar(
                            top_buy_sectors,
                            x='Net Flow (Shares)',
                            y='Sector',
                            orientation='h',
                            text='Net Flow (Shares)',
                            color_discrete_sequence=['green']
                        )
                        fig_sec_buy.update_layout(yaxis={'categoryorder':'total ascending'})
                        fig_sec_buy.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
                        st.plotly_chart(fig_sec_buy, use_container_width=True)
                    else:
                        st.info(f"Tidak ada net buy signifikan oleh {selected_category_for_sector} di sektor manapun.")

                with col_sec_2:
                    st.markdown(f"**Top 10 Sektor Net Sell oleh {selected_category_for_sector}**")
                    top_sell_sectors = df_sector_cat_flow[df_sector_cat_flow['Net Flow (Shares)'] < 0].tail(10).sort_values(by='Net Flow (Shares)')
                    if not top_sell_sectors.empty:
                        fig_sec_sell = px.bar(
                            top_sell_sectors,
                            x='Net Flow (Shares)',
                            y='Sector',
                            orientation='h',
                            text='Net Flow (Shares)',
                            color_discrete_sequence=['red']
                        )
                        fig_sec_sell.update_layout(yaxis={'categoryorder':'total descending'})
                        fig_sec_sell.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
                        st.plotly_chart(fig_sec_sell, use_container_width=True)
                    else:
                        st.info(f"Tidak ada net sell signifikan oleh {selected_category_for_sector} di sektor manapun.")
            else:
                st.info("Tidak ada data aliran dana untuk kategori investor yang dipilih pada tahun ini.")


# --- TAB 3: [DULU TAB 2] ANALISIS SAHAM INDIVIDUAL (MIKRO) ---
with tab3: # Tab berubah dari tab2 menjadi tab3
    st.subheader("Bagaimana Aliran Dana di Satu Saham?")
    
    stocks_in_period = sorted(df_filtered_by_year['Code'].unique())
    
    stock_to_analyze = st.selectbox(
        "Pilih Saham untuk Analisis Mendalam:",
        stocks_in_period,
        index=stocks_in_period.index("BBCA") if "BBCA" in stocks_in_period else 0,
        key="selectbox_stock_analysis"
    )
    
    if stock_to_analyze:
        df_stock_filtered = df_filtered_by_year[df_filtered_by_year['Code'] == stock_to_analyze].sort_values('Date')
        df_state = get_stock_ownership_state(df, stock_to_analyze) 
        
        if df_stock_filtered.empty or df_state.empty:
            st.warning(f"Tidak ada data untuk {stock_to_analyze} pada tahun terpilih.")
        else:
            latest_price = df_stock_filtered.iloc[-1]['Price']
            free_float = df_stock_filtered.iloc[-1]['Free Float']
            stock_sector = df_stock_filtered.iloc[-1]['Sector'] # Ambil sektor
            
            st.markdown(f"**Analisis: {stock_to_analyze} ({stock_sector})**")
            col1, col2, col3 = st.columns(3) # Tambah kolom untuk Sektor
            col1.metric("Harga Terakhir", f"Rp {latest_price:,.0f}" if pd.notna(latest_price) else "N/A")
            col2.metric("Free Float Saham", f"{free_float:.2f}%" if pd.notna(free_float) else "N/A")
            col3.metric("Sektor", stock_sector if pd.notna(stock_sector) else "N/A")

            st.markdown("---")
            
            # [PERUBAHAN] Mengubah Treemap menjadi Pie Charts
            st.markdown("**Peta Kepemilikan (Terbaru)**")
            
            df_local_ownership = df_state[df_state['Tipe'] == 'Local'].sort_values('Jumlah Saham', ascending=False)
            df_foreign_ownership = df_state[df_state['Tipe'] == 'Foreign'].sort_values('Jumlah Saham', ascending=False)
            
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                if not df_local_ownership.empty and df_local_ownership['Jumlah Saham'].sum() > 0:
                    fig_pie_local = px.pie(
                        df_local_ownership,
                        names='Kategori',
                        values='Jumlah Saham',
                        title=f'Komposisi Kepemilikan Lokal {stock_to_analyze}',
                        hole=0.3
                    )
                    fig_pie_local.update_traces(textinfo='percent+label', texttemplate='%{label}: %{value:,.0f} (%{percent})')
                    st.plotly_chart(fig_pie_local, use_container_width=True)
                else:
                    st.info("Tidak ada data kepemilikan lokal.")
            
            with pcol2:
                if not df_foreign_ownership.empty and df_foreign_ownership['Jumlah Saham'].sum() > 0:
                    fig_pie_foreign = px.pie(
                        df_foreign_ownership,
                        names='Kategori',
                        values='Jumlah Saham',
                        title=f'Komposisi Kepemilikan Asing {stock_to_analyze}',
                        hole=0.3
                    )
                    fig_pie_foreign.update_traces(textinfo='percent+label', texttemplate='%{label}: %{value:,.0f} (%{percent})')
                    st.plotly_chart(fig_pie_foreign, use_container_width=True)
                else:
                    st.info("Tidak ada data kepemilikan asing.")

            st.markdown("---")
            st.markdown("**Detail Rotasi Kepemilikan per Periode**")
            # [PERUBAHAN] Ganti Chart Rotasi dengan Tabel Detail
            cols_to_display_detail = [
                'Date', 'Top_Buyer', 'Top_Buyer_Vol', 'Top_Seller', 'Top_Seller_Vol',
                'Price', 'Price_Chg %'
            ]
            
            df_stock_detail = df_stock_filtered[cols_to_display_detail].copy()
            
            # Format angka untuk tampilan
            df_stock_detail['Top_Buyer_Vol'] = df_stock_detail['Top_Buyer_Vol'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
            df_stock_detail['Top_Seller_Vol'] = df_stock_detail['Top_Seller_Vol'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
            df_stock_detail['Price'] = df_stock_detail['Price'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
            
            st.dataframe(
                df_stock_detail,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Tanggal", format="DD-MM-YYYY"),
                    "Top_Buyer": st.column_config.TextColumn("Top Buyer"),
                    "Top_Buyer_Vol": st.column_config.TextColumn("Vol Buyer"),
                    "Top_Seller": st.column_config.TextColumn("Top Seller"),
                    "Top_Seller_Vol": st.column_config.TextColumn("Vol Seller"),
                    "Price": st.column_config.TextColumn("Harga"),
                    "Price_Chg %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                }
            )


# --- TAB 4: [DULU TAB 3] SCREENER ROTASI ---
with tab4: # Tab berubah dari tab3 menjadi tab4
    st.subheader("Screener Rotasi Kepemilikan")
    st.info("Gunakan filter di sidebar (Filter Tahun & Filter Screener) untuk mencari rotasi spesifik.")
    
    cols_to_display = [
        'Date', 'Code', 'Sector', 'Top_Buyer', 'Top_Buyer_Vol', 'Top_Seller', 'Top_Seller_Vol',
        'Price', 'Price_Chg %', 'Free Float'
    ]
    # Pastikan kolom 'Sector' ada sebelum diakses
    if 'Sector' in df_screener_filtered.columns:
        df_screener = df_screener_filtered[cols_to_display].sort_values(by=['Date', 'Top_Buyer_Vol'], ascending=[False, False])
    else:
        # Jika 'Sector' tidak ada, hapus dari cols_to_display dan berikan warning
        st.warning("Kolom 'Sector' tidak ditemukan untuk screener. Pastikan file olahan KSEI memiliki data sektor.")
        cols_to_display.remove('Sector')
        df_screener = df_screener_filtered[cols_to_display].sort_values(by=['Date', 'Top_Buyer_Vol'], ascending=[False, False])


    df_screener_display = df_screener.copy()
    
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
            "Sector": st.column_config.TextColumn("Sektor"), # Tambahkan kolom Sektor
            "Top_Buyer": st.column_config.TextColumn("Top Buyer"),
            "Top_Buyer_Vol": st.column_config.TextColumn("Vol Buyer"),
            "Top_Seller": st.column_config.TextColumn("Top Seller"),
            "Top_Seller_Vol": st.column_config.TextColumn("Vol Seller"),
            "Price": st.column_config.TextColumn("Harga"),
            "Price_Chg %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
            "Free Float": st.column_config.NumberColumn("Free Float %", format="%.2f%%")
        }
    )

