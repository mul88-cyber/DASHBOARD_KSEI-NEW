# ==============================================================================
# 📦 1) IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots # Tidak digunakan lagi di Tab 3
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
        msg = "❌ Gagal otentikasi: 'st.secrets' tidak menemukan key [gcp_service_account]."
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

        if 'Sector' in df.columns:
            df['Sector'] = df['Sector'].astype(str).str.strip().fillna('Others')
        else:
            df['Sector'] = 'Others'

        cols_to_numeric = [
            'Price', 'Price_Chg %', 'Free Float', 'Total_Local', 'Total_Foreign',
            'Top_Buyer_Vol', 'Top_Seller_Vol', 'Sec. Num'
        ] + OWNERSHIP_COLS + OWNERSHIP_CHG_COLS

        for col in cols_to_numeric:
            if col in df.columns:
                cleaned_col = df[col].astype(str).str.strip()
                cleaned_col = cleaned_col.str.replace(',', '', regex=False)
                if col == 'Sec. Num' and cleaned_col.eq('').any():
                    # Tidak perlu warning, cukup isi 0
                    pass
                df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0)
            elif col == 'Sec. Num':
                 st.error("Kolom 'Sec. Num' tidak ditemukan di file CSV. Pie chart Non-Free Float tidak dapat dibuat.", icon="🚨")
                 df['Sec. Num'] = 0

        df = df.dropna(subset=['Date', 'Code'])

        local_chg_cols = [col for col in OWNERSHIP_CHG_COLS if 'Local' in col]
        foreign_chg_cols = [col for col in OWNERSHIP_CHG_COLS if 'Foreign' in col]

        df['Total_Local_chg'] = df[local_chg_cols].sum(axis=1)
        df['Total_Foreign_chg'] = df[foreign_chg_cols].sum(axis=1)
        df['Total_chg'] = df['Total_Local_chg'] + df['Total_Foreign_chg']

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
    net_flow = df_filtered_by_year[OWNERSHIP_CHG_COLS].sum().reset_index()
    net_flow.columns = ['Kategori', 'Total Net Flow (Shares)']
    net_flow['Kategori'] = net_flow['Kategori'].str.replace('_chg', '')
    net_flow = net_flow.sort_values(by='Total Net Flow (Shares)', ascending=False)

    cum_flow = df_filtered_by_year.groupby('Date')[['Total_Local_chg', 'Total_Foreign_chg']].sum().cumsum().reset_index()
    cum_flow = cum_flow.melt('Date', var_name='Kategori', value_name='Cumulative Flow')
    cum_flow['Kategori'] = cum_flow['Kategori'].str.replace('_chg', ' (Net)')

    return net_flow, cum_flow

@st.cache_data
def calculate_sector_rotation(df_filtered_by_year, selected_category):
    """(TAB 2) Menghitung aliran dana bersih kategori tertentu per sektor."""
    if 'Sector' not in df_filtered_by_year.columns or df_filtered_by_year['Sector'].nunique() <= 1:
        return pd.DataFrame(), "Data sektor tidak tersedia atau hanya 'Others'."
    category_chg_col = f"{selected_category}_chg"
    if category_chg_col not in df_filtered_by_year.columns:
        return pd.DataFrame(), f"Kolom '{category_chg_col}' tidak ditemukan."
    sector_category_flow = df_filtered_by_year.groupby('Sector')[category_chg_col].sum().reset_index()
    sector_category_flow.columns = ['Sector', 'Net Flow (Shares)']
    sector_category_flow = sector_category_flow.sort_values(by='Net Flow (Shares)', ascending=False)
    return sector_category_flow, None

@st.cache_data
def calculate_monthly_sector_flow(df_filtered_by_year):
    """(TAB 4 Chart) Menghitung total aliran dana bersih bulanan per sektor."""
    if 'Sector' not in df_filtered_by_year.columns or df_filtered_by_year['Sector'].nunique() <= 1:
        return pd.DataFrame(), "Data sektor tidak tersedia."
    df_temp = df_filtered_by_year.set_index('Date')
    monthly_sector_flow = df_temp.groupby('Sector').resample('MS')['Total_chg'].sum().reset_index()
    monthly_sector_flow.columns = ['Sector', 'Month', 'Net Flow (Shares)']
    return monthly_sector_flow, None

@st.cache_data
def get_stock_ownership_state(df, stock_code):
    """(TAB 3 Pie) Mengambil data kepemilikan TERBARU untuk 1 saham."""
    df_stock = df[df['Code'] == stock_code]
    if df_stock.empty:
        return pd.DataFrame(), pd.Series(dtype='object')

    latest_row = df_stock.sort_values('Date').iloc[-1]
    df_state = latest_row[OWNERSHIP_COLS].reset_index()
    df_state.columns = ['Kategori', 'Jumlah Saham']

    total_shares_pie1 = df_state['Jumlah Saham'].sum()
    if total_shares_pie1 > 0:
        df_state['Persentase'] = (df_state['Jumlah Saham'] / total_shares_pie1) * 100
    else:
        df_state['Persentase'] = 0

    return df_state.sort_values(by='Jumlah Saham', ascending=False), latest_row

@st.cache_data
def calculate_monthly_shareholder_change_table(df_stock_filtered):
    """(TAB 3 Table) Menghitung perubahan bulanan per kategori shareholder."""
    if df_stock_filtered.empty:
        return pd.DataFrame()
    df_temp = df_stock_filtered.set_index('Date')
    monthly_changes = df_temp.resample('MS')[OWNERSHIP_CHG_COLS].sum()
    monthly_changes.columns = [col.replace('_chg', '') for col in OWNERSHIP_CHG_COLS]
    monthly_changes = monthly_changes.sort_index(ascending=False)
    monthly_changes = monthly_changes.reset_index()
    monthly_changes.rename(columns={'Date': 'Month'}, inplace=True)
    return monthly_changes

# [BARU] Fungsi untuk line chart histori kepemilikan (TAB 3)
@st.cache_data
def calculate_historical_ownership_pct(df_stock_filtered):
    """(TAB 3 Line Chart) Menghitung persentase kepemilikan historis per kategori."""
    if df_stock_filtered.empty or not all(col in df_stock_filtered.columns for col in OWNERSHIP_COLS):
        return pd.DataFrame()

    df_hist = df_stock_filtered[['Date'] + OWNERSHIP_COLS].copy()
    # Hitung total saham per tanggal
    df_hist['Total_Shares_Calc'] = df_hist[OWNERSHIP_COLS].sum(axis=1)

    # Hitung persentase untuk setiap kategori
    for col in OWNERSHIP_COLS:
        # Handle jika Total_Shares_Calc == 0
        df_hist[f'{col}_pct'] = np.where(df_hist['Total_Shares_Calc'] > 0, (df_hist[col] / df_hist['Total_Shares_Calc']) * 100, 0)

    # Pilih kolom persentase saja
    pct_cols = [f'{col}_pct' for col in OWNERSHIP_COLS]
    df_hist_pct = df_hist[['Date'] + pct_cols]

    # Melt untuk format plotting
    df_melted = df_hist_pct.melt(id_vars=['Date'], var_name='Kategori_pct', value_name='Persentase')
    df_melted['Kategori'] = df_melted['Kategori_pct'].str.replace('_pct', '')

    # Filter kategori yang selalu 0%
    total_pct_per_cat = df_melted.groupby('Kategori')['Persentase'].sum()
    active_categories = total_pct_per_cat[total_pct_per_cat > 0.01].index # Toleransi kecil
    df_melted = df_melted[df_melted['Kategori'].isin(active_categories)]


    return df_melted[['Date', 'Kategori', 'Persentase']]


def highlight_max_min(s):
    '''Highlight maximum (positive) in green and minimum (negative) in red.'''
    # Pastikan input adalah numerik
    s_numeric = pd.to_numeric(s, errors='coerce')
    max_val = s_numeric[s_numeric > 0].max()
    min_val = s_numeric[s_numeric < 0].min()
    colors = []
    for val in s_numeric:
        if pd.notna(val):
            if val == max_val and val > 0:
                colors.append('background-color: lightgreen')
            elif val == min_val and val < 0:
                colors.append('background-color: lightcoral')
            else:
                colors.append('')
        else:
             colors.append('')
    return colors


# ==============================================================================
# 💎 5) LAYOUT UTAMA (HEADER)
# ==============================================================================
st.title("🌊 Dashboard Analisis Aliran Dana KSEI")
st.caption("Menganalisis rotasi kepemilikan saham (flow) untuk mengambil keputusan.")

df, status_msg, status_level = load_data()

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
st.sidebar.header("Filter Screener (u/ Tab 4)") # Nomor Tab diupdate

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
    df_screener_filtered = df_screener_filtered[df_screener_filtered['Top_Buyer'].isin(selected_buyers)]
if selected_sellers:
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
    "📊 **Analisis Sektor (Rotasi)**",
    "📈 **Analisa Individual**", # Nama Tab diubah
    "🔍 **Screener Rotasi**"
])

# --- TAB 1: RINGKASAN ALIRAN DANA (MARKET) ---
# ... (Kode Tab 1 tidak berubah) ...
with tab1:
    st.subheader(f"Peta Aliran Dana Market (Tahun: {', '.join(map(str, selected_years))})")
    df_net_flow, df_cum_flow = calculate_macro_flow(df_filtered_by_year)
    st.markdown("**Aliran Dana Kumulatif (Lokal vs Asing)**")
    fig_macro = px.line(df_cum_flow, x='Date', y='Cumulative Flow', color='Kategori', title='Aliran Kumulatif Lokal vs Asing (Total Market)', labels={'Cumulative Flow': 'Total Saham (Kumulatif)', 'Date': 'Tanggal'})
    fig_macro.update_traces(hovertemplate='Tanggal: %{x|%d %b %Y}<br>Flow: %{y:,.0f}<extra></extra>')
    fig_macro.update_layout(hovermode="x unified")
    st.plotly_chart(fig_macro, use_container_width=True)
    st.markdown("---")
    st.markdown("**Kategori Investor Terkuat (Net Flow)**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 5 Kategori Net Buy**")
        top_5_buy = df_net_flow.head(5)
        fig_buy = px.bar(top_5_buy, x='Total Net Flow (Shares)', y='Kategori', orientation='h', text='Total Net Flow (Shares)')
        fig_buy.update_layout(yaxis={'categoryorder':'total ascending'})
        fig_buy.update_traces(texttemplate='%{x:,.0f}', textposition='outside', marker_color='green', hovertemplate='Kategori: %{y}<br>Net Flow: %{x:,.0f}<extra></extra>')
        st.plotly_chart(fig_buy, use_container_width=True)
    with col2:
        st.markdown("**Top 5 Kategori Net Sell**")
        top_5_sell = df_net_flow.tail(5).sort_values(by='Total Net Flow (Shares)')
        fig_sell = px.bar(top_5_sell, x='Total Net Flow (Shares)', y='Kategori', orientation='h', text='Total Net Flow (Shares)')
        fig_sell.update_layout(yaxis={'categoryorder':'total descending'})
        fig_sell.update_traces(texttemplate='%{x:,.0f}', textposition='outside', marker_color='red', hovertemplate='Kategori: %{y}<br>Net Flow: %{x:,.0f}<extra></extra>')
        st.plotly_chart(fig_sell, use_container_width=True)

# --- TAB 2: ANALISIS SEKTOR (ROTASI) ---
# ... (Kode Tab 2 tidak berubah) ...
with tab2:
    st.subheader(f"Analisis Rotasi Kategori Investor per Sektor (Tahun: {', '.join(map(str, selected_years))})")
    if 'Sector' not in df_filtered_by_year.columns or df_filtered_by_year['Sector'].nunique() <= 1:
        st.warning("Kolom 'Sector' tidak ditemukan atau hanya berisi 'Others'.")
    else:
        all_categories_for_sector = sorted([col.replace('_chg', '') for col in OWNERSHIP_CHG_COLS])
        selected_category_for_sector = st.selectbox("Pilih Kategori Investor:", all_categories_for_sector, key="sector_category_select")
        if selected_category_for_sector:
            df_sector_cat_flow, error_sec_cat = calculate_sector_rotation(df_filtered_by_year, selected_category_for_sector)
            if error_sec_cat: st.error(error_sec_cat)
            elif not df_sector_cat_flow.empty:
                st.markdown(f"**Net Flow ({selected_category_for_sector}) per Sektor**")
                col_sec_1, col_sec_2 = st.columns(2)
                with col_sec_1:
                    st.markdown(f"**Top 10 Sektor Net Buy**")
                    top_buy_sectors = df_sector_cat_flow[df_sector_cat_flow['Net Flow (Shares)'] > 0].head(10)
                    if not top_buy_sectors.empty:
                        fig_sec_buy = px.bar(top_buy_sectors, x='Net Flow (Shares)', y='Sector', orientation='h', text='Net Flow (Shares)', color_discrete_sequence=['green'])
                        fig_sec_buy.update_layout(yaxis={'categoryorder':'total ascending'})
                        fig_sec_buy.update_traces(texttemplate='%{x:,.0f}', textposition='outside', hovertemplate='Sektor: %{y}<br>Net Flow: %{x:,.0f}<extra></extra>')
                        st.plotly_chart(fig_sec_buy, use_container_width=True)
                    else: st.info(f"Tidak ada net buy signifikan.")
                with col_sec_2:
                    st.markdown(f"**Top 10 Sektor Net Sell**")
                    top_sell_sectors = df_sector_cat_flow[df_sector_cat_flow['Net Flow (Shares)'] < 0].tail(10).sort_values(by='Net Flow (Shares)')
                    if not top_sell_sectors.empty:
                        fig_sec_sell = px.bar(top_sell_sectors, x='Net Flow (Shares)', y='Sector', orientation='h', text='Net Flow (Shares)', color_discrete_sequence=['red'])
                        fig_sec_sell.update_layout(yaxis={'categoryorder':'total descending'})
                        fig_sec_sell.update_traces(texttemplate='%{x:,.0f}', textposition='outside', hovertemplate='Sektor: %{y}<br>Net Flow: %{x:,.0f}<extra></extra>')
                        st.plotly_chart(fig_sec_sell, use_container_width=True)
                    else: st.info(f"Tidak ada net sell signifikan.")
            else: st.info("Tidak ada data aliran dana.")

# --- TAB 3: ANALISA INDIVIDUAL ---
with tab3:
    st.subheader("Bagaimana Aliran Dana di Satu Saham?")
    stocks_in_period = sorted(df_filtered_by_year['Code'].unique())
    stock_to_analyze = st.selectbox("Pilih Saham:", stocks_in_period, index=stocks_in_period.index("BBCA") if "BBCA" in stocks_in_period else 0, key="selectbox_stock_analysis")

    if stock_to_analyze:
        df_stock_filtered = df_filtered_by_year[df_filtered_by_year['Code'] == stock_to_analyze].sort_values('Date')
        df_state, latest_row_data = get_stock_ownership_state(df, stock_to_analyze)

        if df_stock_filtered.empty or df_state.empty:
            st.warning(f"Tidak ada data untuk {stock_to_analyze} pada tahun terpilih.")
        else:
            latest_price = latest_row_data.get('Price', np.nan)
            free_float = latest_row_data.get('Free Float', np.nan)
            stock_sector = latest_row_data.get('Sector', 'N/A')
            sec_num = latest_row_data.get('Sec. Num', 0)
            total_local = latest_row_data.get('Total_Local', 0)
            total_foreign = latest_row_data.get('Total_Foreign', 0)

            st.markdown(f"**Analisis: {stock_to_analyze} ({stock_sector})**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Harga Terakhir", f"Rp {latest_price:,.0f}" if pd.notna(latest_price) else "N/A")
            col2.metric("Free Float Saham", f"{free_float:.2f}%" if pd.notna(free_float) else "N/A")
            col3.metric("Sektor", stock_sector if pd.notna(stock_sector) else "N/A")
            st.markdown("---")

            # --- [PERUBAHAN] Layout Pie Charts & Line Chart ---
            st.markdown("**Peta Kepemilikan (Terbaru) & Tren Historis**")
            # Bagi layout menjadi 3 kolom: Pie1, Pie2, Line
            pcol1, pcol2, pcol3 = st.columns([1, 1, 2]) # Kolom ke-3 lebih lebar

            with pcol1:
                # Pie Chart 1: All Categories
                if not df_state.empty and df_state['Jumlah Saham'].sum() > 0:
                    fig_pie_all = px.pie(
                        df_state, names='Kategori', values='Jumlah Saham',
                        title=f'Komposisi Semua Holder', hole=0.3 )
                    fig_pie_all.update_traces(textinfo='percent+label', texttemplate='%{label}(%{percent})', sort=False, showlegend=False) # Hide legend
                    fig_pie_all.update_layout(margin=dict(l=20, r=20, t=30, b=20)) # Margin kecil
                    st.plotly_chart(fig_pie_all, use_container_width=True)
                else: st.info("No ownership data.")

            with pcol2:
                # Pie Chart 2: Local / Foreign / Non-FF
                if sec_num > 0:
                    non_free_float_val = max(0, sec_num - total_local - total_foreign)
                    df_pie_dist = pd.DataFrame({'Tipe': ['Lokal', 'Asing', 'Non Publik'],
                                                'Jumlah Saham': [total_local, total_foreign, non_free_float_val]})
                    df_pie_dist = df_pie_dist[df_pie_dist['Jumlah Saham'] > 0]
                    if not df_pie_dist.empty:
                        fig_pie_dist = px.pie(
                            df_pie_dist, names='Tipe', values='Jumlah Saham',
                            title=f'Distribusi Umum', hole=0.3 )
                        fig_pie_dist.update_traces(textinfo='percent+label', texttemplate='%{label}(%{percent})', sort=False, showlegend=False) # Hide legend
                        fig_pie_dist.update_layout(margin=dict(l=20, r=20, t=30, b=20)) # Margin kecil
                        st.plotly_chart(fig_pie_dist, use_container_width=True)
                    else: st.info("No distribution data.")
                else: st.warning("'Sec. Num' invalid.")

            with pcol3:
                # [BARU] Line Chart Historical Ownership %
                st.markdown("**Tren Kepemilikan Historis (%)**")
                df_hist_pct = calculate_historical_ownership_pct(df_stock_filtered)
                if not df_hist_pct.empty:
                    fig_hist_pct = px.line(
                        df_hist_pct,
                        x='Date',
                        y='Persentase',
                        color='Kategori',
                        title=f'Tren Persentase Kepemilikan {stock_to_analyze}',
                        labels={'Date': 'Tanggal', 'Persentase': '% Kepemilikan'}
                    )
                    fig_hist_pct.update_layout(hovermode='x unified', yaxis_ticksuffix='%')
                    fig_hist_pct.update_traces(hovertemplate='Tgl: %{x|%d%b%y}<br>%{fullData.name}: %{y:.2f}%<extra></extra>')
                    st.plotly_chart(fig_hist_pct, use_container_width=True)
                else:
                    st.warning("Tidak ada data historis kepemilikan untuk ditampilkan.")


            st.markdown("---")
            # [PERUBAHAN] Tabel Detail Bulanan (Layout tidak berubah)
            st.markdown("**Detail Rotasi Kepemilikan per Bulan**")
            df_monthly_change = calculate_monthly_shareholder_change_table(df_stock_filtered)

            if not df_monthly_change.empty:
                df_display_monthly = df_monthly_change.copy()
                df_display_monthly['Month'] = df_display_monthly['Month'].dt.strftime('%b %Y')
                numeric_cols_to_style = df_display_monthly.columns.drop('Month')

                # Tampilkan tabel (Pastikan use_container_width=True)
                st.dataframe(
                    df_display_monthly.style.apply(highlight_max_min, subset=numeric_cols_to_style, axis=1)
                                          .format("{:,.0f}", subset=numeric_cols_to_style, na_rep='0'),
                    use_container_width=True, # <-- PENTING UNTUK FULL WIDTH
                    hide_index=True
                )
            else:
                st.warning("Tidak ada data perubahan bulanan untuk ditampilkan.")


# --- TAB 4: SCREENER ROTASI ---
with tab4:
    st.subheader("Screener Rotasi Kepemilikan")
    # ... (Kode Chart Aliran Sektor Bulanan tidak berubah) ...
    st.markdown("**Tren Aliran Dana Bersih Bulanan per Sektor**")
    df_monthly_sec_flow, error_monthly_sec = calculate_monthly_sector_flow(df_filtered_by_year)
    if error_monthly_sec: st.warning(error_monthly_sec)
    elif not df_monthly_sec_flow.empty:
        total_abs_flow = df_monthly_sec_flow.groupby('Sector')['Net Flow (Shares)'].apply(lambda x: x.abs().sum()).nlargest(10).index
        df_monthly_sec_flow_top = df_monthly_sec_flow[df_monthly_sec_flow['Sector'].isin(total_abs_flow)]
        fig_monthly_sec = px.line(df_monthly_sec_flow_top, x='Month', y='Net Flow (Shares)', color='Sector', title='Tren Aliran Dana Bersih Bulanan (Top 10 Sektor)', labels={'Month': 'Bulan', 'Net Flow (Shares)': 'Net Flow Bulanan (Saham)'}, markers=True)
        fig_monthly_sec.update_layout(hovermode='x unified')
        fig_monthly_sec.update_traces(hovertemplate='Bulan: %{x|%b %Y}<br>Sektor: %{fullData.name}<br>Flow: %{y:,.0f}<extra></extra>')
        st.plotly_chart(fig_monthly_sec, use_container_width=True)
    else: st.info("Tidak ada data aliran dana sektoral bulanan.")

    st.markdown("---")
    st.info("Gunakan filter di sidebar (Filter Tahun & Filter Screener) untuk mencari rotasi spesifik di tabel bawah.")
    cols_to_display = ['Date', 'Code', 'Sector', 'Top_Buyer', 'Top_Buyer_Vol', 'Top_Seller', 'Top_Seller_Vol', 'Price', 'Price_Chg %', 'Free Float']
    use_cols = cols_to_display[:]
    if 'Sector' not in df_screener_filtered.columns:
        st.warning("Kolom 'Sector' tidak ditemukan untuk screener.")
        use_cols.remove('Sector')
    df_screener = df_screener_filtered[use_cols].sort_values(by=['Date', 'Top_Buyer_Vol'], ascending=[False, False])
    df_screener_display = df_screener.copy()
    df_screener_display['Top_Buyer_Vol'] = df_screener_display['Top_Buyer_Vol'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
    df_screener_display['Top_Seller_Vol'] = df_screener_display['Top_Seller_Vol'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
    df_screener_display['Price'] = df_screener_display['Price'].apply(lambda x: f"Rp {x:,.0f}" if pd.notna(x) else 'N/A')
    col_config_screener = {
        "Date": st.column_config.DateColumn("Tanggal", format="DD-MM-YYYY"), "Code": "Saham",
        "Top_Buyer": "Top Buyer", "Top_Buyer_Vol": "Vol Buyer", "Top_Seller": "Top Seller", "Top_Seller_Vol": "Vol Seller",
        "Price": "Harga", "Price_Chg %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
        "Free Float": st.column_config.NumberColumn("Free Float %", format="%.2f%%")
    }
    if 'Sector' in use_cols: col_config_screener["Sector"] = "Sektor"
    st.dataframe(df_screener_display, use_container_width=True, hide_index=True, column_config=col_config_screener)

