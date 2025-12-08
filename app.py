# ==============================================================================
# 📦 1) IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    page_title="🌊 Dashboard Bandarmology KSEI (Bulanan)",
    layout="wide",
    page_icon="🌊"
)

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
                df[col] = pd.to_numeric(cleaned_col, errors='coerce').fillna(0)

        df = df.dropna(subset=['Date', 'Code'])

        local_chg_rp_cols = [col for col in OWNERSHIP_CHG_RP_COLS if 'Local' in col and col in df.columns]
        foreign_chg_rp_cols = [col for col in OWNERSHIP_CHG_RP_COLS if 'Foreign' in col and col in df.columns]
        df['Total_Local_chg_Rp'] = df[local_chg_rp_cols].sum(axis=1)
        df['Total_Foreign_chg_Rp'] = df[foreign_chg_rp_cols].sum(axis=1)
        df['Total_chg_Rp'] = df['Total_Local_chg_Rp'] + df['Total_Foreign_chg_Rp']

        msg = f"Data KSEI (2025-Now) berhasil dimuat (file ID: {file_id})."
        return df, msg, "success"

    except Exception as e:
        msg = f"❌ Terjadi error saat memuat data KSEI: {e}."
        return pd.DataFrame(), msg, "error"

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
def calculate_smart_money_signals(df_year, window_periods=3):
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
        if sm_flow_sum > 5_000_000_000 and retail_flow_sum < 0: status = "🔥 Big Accumulation"; score = 100
        elif sm_flow_sum > 2_000_000_000 and price_chg_pct <= 3: status = "💎 Divergence (Collect)"; score = 80
        elif sm_flow_sum < -5_000_000_000 and retail_flow_sum > 0: status = "⚠️ Distribution"; score = -50
        if abs(score) >= 50:
            results.append({'Code': code, 'Sector': df_window.iloc[-1].get('Sector', 'N/A'), 'Price': last_price, 'Price Chg (Window)%': price_chg_pct, 'Smart Money Flow (Rp)': sm_flow_sum, 'Retail Flow (Rp)': retail_flow_sum, 'Signal': status, 'Score': score})
    df_res = pd.DataFrame(results)
    if not df_res.empty: df_res = df_res.sort_values(by='Smart Money Flow (Rp)', ascending=False)
    return df_res

# --- FUNGSI SANKEY CHART (UPDATED WITH LABELS) ---
def create_sankey_chart(df, stock_code, selected_date, mode='Volume'):
    """Membuat diagram Sankey dengan label angka (Jt/M/T)."""
    
    # 1. Filter Data
    row = df[(df['Code'] == stock_code) & (df['Date'] == selected_date)]
    if row.empty: return None
    row = row.iloc[0]
    
    # 2. Tentukan Kolom & Satuan
    cols_to_use = OWNERSHIP_CHG_COLS if mode == 'Volume' else OWNERSHIP_CHG_RP_COLS
    is_rp = (mode == 'Value')
    
    # 3. Pisahkan Seller & Buyer
    sellers = []; buyers = []; total_vol = 0
    
    for col in cols_to_use:
        val = row[col]
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
            
    if not sellers and not buyers: return None
    
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
    s_numeric = pd.to_numeric(s, errors='coerce')
    max_val = s_numeric[s_numeric > 0].max()
    min_val = s_numeric[s_numeric < 0].min()
    colors = []
    for val in s_numeric:
        if pd.notna(val):
            if val == max_val and val > 0: colors.append('background-color: lightgreen')
            elif val == min_val and val < 0: colors.append('background-color: lightcoral')
            else: colors.append('')
        else: colors.append('')
    return colors

# ==============================================================================
# 💎 5) LAYOUT UTAMA
# ==============================================================================
st.title("🌊 Dashboard Bandarmology KSEI (Data Bulanan)")
st.caption("Analisis khusus data 2025 ke atas.")

df, status_msg, status_level = load_data()

if status_level == "success":
    st.toast(status_msg, icon="✅")
elif status_level == "error":
    st.error(status_msg)
    st.stop()

# ==============================================================================
# 🧭 SIDEBAR
# ==============================================================================
st.sidebar.header("🎛️ Filter & Navigasi")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info(f"📅 Data Range: {df['Date'].dt.date.min()} s/d {df['Date'].dt.date.max()}")

# Filter Screener (Tab 4)
st.sidebar.header("Filter Screener (Tab 4)")
min_rotation_value = st.sidebar.number_input(
    "Min. Value Rotasi (Rp)", 
    value=1_000_000_000, 
    step=1_000_000_000, 
    format="%d"
)
st.sidebar.caption(f"Setting saat ini: Rp {min_rotation_value:,.0f}")

# ==============================================================================
# 📑 TABS VISUALISASI
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌊 **Makro (Market)**",
    "📊 **Analisis Sektor**",
    "📈 **Analisa Individual**", 
    "🔍 **Screener Rotasi**",
    "💎 **Sinyal Potensial (AI)**"
])

# --- TAB 1: MAKRO ---
with tab1:
    st.subheader("Peta Aliran Dana Market (2025)")
    df_net_flow, df_cum_flow = calculate_macro_flow(df)
    
    st.markdown("**Aliran Dana Kumulatif (Rp)**")
    fig_macro = px.line(df_cum_flow, x='Date', y='Cumulative Flow (Rp)', color='Kategori', title='Akumulasi Net Flow Lokal vs Asing')
    # Update format Y axis ke koma ribuan
    fig_macro.update_layout(hovermode="x unified", yaxis_tickformat=',.0f')
    fig_macro.update_traces(hovertemplate='Tanggal: %{x}<br>Flow: %{y:,.0f}')
    st.plotly_chart(fig_macro, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Net Buy (Kategori Investor)**")
        st.bar_chart(df_net_flow.head(5).set_index('Kategori')['Total Net Flow (Rp)'], color="#00FF00")
    with c2:
        st.markdown("**Top Net Sell (Kategori Investor)**")
        st.bar_chart(df_net_flow.tail(5).set_index('Kategori')['Total Net Flow (Rp)'].sort_values(), color="#FF0000")

# --- TAB 2: SEKTOR ---
with tab2:
    st.subheader("Rotasi Sektor (2025)")
    cats = sorted([c.replace('_chg_Rp','') for c in OWNERSHIP_CHG_RP_COLS])
    sel_cat = st.selectbox("Pilih Kategori Investor:", cats, index=cats.index('Foreign IB') if 'Foreign IB' in cats else 0)
    
    df_sec_flow, msg = calculate_sector_rotation(df, sel_cat)
    if not df_sec_flow.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Top Buy Sektor oleh {sel_cat}**")
            fig_buy = px.bar(df_sec_flow.head(10), x='Net Flow (Rp)', y='Sector', orientation='h', color_discrete_sequence=['green'], text_auto=',.0f')
            fig_buy.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_tickformat=',.0f')
            st.plotly_chart(fig_buy, use_container_width=True)
        with c2:
            st.markdown(f"**Top Sell Sektor oleh {sel_cat}**")
            fig_sell = px.bar(df_sec_flow.tail(10), x='Net Flow (Rp)', y='Sector', orientation='h', color_discrete_sequence=['red'], text_auto=',.0f')
            fig_sell.update_layout(yaxis={'categoryorder':'total descending'}, xaxis_tickformat=',.0f')
            st.plotly_chart(fig_sell, use_container_width=True)

# --- TAB 3: INDIVIDUAL (SUNBURST & SANKEY FIX) ---
with tab3:
    st.subheader("Deep Dive Saham")
    stocks = sorted(df['Code'].unique())
    sel_stock = st.selectbox("Pilih Saham:", stocks, index=stocks.index('BBRI') if 'BBRI' in stocks else 0)
    
    if sel_stock:
        df_stock = df[df['Code'] == sel_stock].sort_values('Date')
        df_state, last_row = get_stock_ownership_state(df, sel_stock)
        
        # 1. Metrics (Top)
        c1, c2, c3 = st.columns(3)
        c1.metric("Harga Terakhir", f"Rp {last_row['Price']:,.0f}")
        c2.metric("Free Float", f"{last_row['Free Float']}%")
        c3.metric("Sektor", last_row['Sector'])
        
        st.markdown("---")
        
        # 2. Distribution Layout (Sunburst + Breakdown)
        c_sun, c_bar = st.columns([1.5, 1])
        with c_sun:
            st.markdown("**Peta Kepemilikan (Sunburst)**")
            # Logic Sunburst dengan Format
            sec_num = last_row.get('Sec. Num', 0)
            total_local = last_row.get('Total_Local', 0)
            total_foreign = last_row.get('Total_Foreign', 0)
            gap_shares = max(0, sec_num - total_local - total_foreign)
            
            # --- LABEL FORMATTING (Jt/M/T) ---
            lbl_total = f"Total\n({format_id_short(sec_num)})"
            lbl_local = f"Lokal\n({format_id_short(total_local)})"
            lbl_foreign = f"Asing\n({format_id_short(total_foreign)})"
            
            labels = [lbl_total, lbl_local, lbl_foreign]
            parents = ["", lbl_total, lbl_total] # Parent refers to existing label string!
            values = [sec_num, total_local, total_foreign]
            
            if gap_shares > 0: 
                lbl_gap = f"Warkat\n({format_id_short(gap_shares)})"
                labels.append(lbl_gap)
                parents.append(lbl_total)
                values.append(gap_shares)
            
            for index, row in df_state.iterrows():
                if row['Jumlah Saham'] > 0:
                    cat_name = row['Kategori']
                    # Tentukan Parent berdasarkan nama kategori asli
                    parent_node = lbl_local if "Local" in cat_name else lbl_foreign
                    
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
                insidetextorientation='radial'
            ))
            fig_sun.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=450)
            st.plotly_chart(fig_sun, use_container_width=True)
            
        with c_bar:
            st.markdown("**Top 5 Holders (Volume)**")
            top_h = df_state.sort_values('Jumlah Saham', ascending=False).head(5).copy()
            # Apply formatter
            top_h['Label'] = top_h['Jumlah Saham'].apply(lambda x: format_id_short(x))
            
            fig_hbar = px.bar(top_h, x='Jumlah Saham', y='Kategori', orientation='h', text='Label')
            fig_hbar.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
            st.plotly_chart(fig_hbar, use_container_width=True)

        st.markdown("---")
        
        # 3. SANKEY FLOW
        st.subheader("🔄 Visualisasi Aliran Dana (Sankey Flow)")
        col_f1, col_f2 = st.columns([1, 4])
        with col_f1:
            st.markdown("##### Filter Flow")
            available_dates = df_stock['Date'].sort_values(ascending=False).dt.strftime('%Y-%m-%d').unique()
            sel_date_str = st.selectbox("Pilih Bulan:", available_dates, index=0)
            sel_date_ts = pd.Timestamp(sel_date_str)
            mode_sankey = st.radio("Satuan:", ["Value (Rp)", "Volume (Lembar)"])
            mode_key = 'Value' if 'Rp' in mode_sankey else 'Volume'
        with col_f2:
            fig_sankey = create_sankey_chart(df, sel_stock, sel_date_ts, mode=mode_key)
            if fig_sankey:
                st.plotly_chart(fig_sankey, use_container_width=True)
            else:
                st.info(f"Tidak ada perubahan kepemilikan signifikan pada bulan {sel_date_str}.")
        
        st.markdown("---")
        
        # 4. Table (Bottom)
        st.markdown("### 📅 Detail Perubahan Bulanan (Volume Lembar)")
        df_m_chg = calculate_monthly_change_table(df_stock)
        st.dataframe(
            df_m_chg.style
            .apply(highlight_max_min, subset=OWNERSHIP_CHG_COLS, axis=1)
            .format("{:,.0f}", subset=OWNERSHIP_CHG_COLS), 
            use_container_width=True, 
            hide_index=True
        )

# --- TAB 4: SCREENER (FIXED) ---
with tab4:
    st.subheader("Screener Big Rotation")
    df_monthly_sec, _ = calculate_monthly_sector_flow(df)
    if not df_monthly_sec.empty:
        st.markdown("**Tren Flow Sektor Bulanan**")
        top_sectors = df_monthly_sec.groupby('Sector')['Net Flow (Rp)'].sum().abs().nlargest(5).index
        fig_sec = px.line(df_monthly_sec[df_monthly_sec['Sector'].isin(top_sectors)], x='Month', y='Net Flow (Rp)', color='Sector')
        fig_sec.update_layout(yaxis_tickformat=',.0f')
        st.plotly_chart(fig_sec, use_container_width=True)

    st.markdown(f"**Saham dengan Rotasi > Rp {min_rotation_value:,.0f}**")
    
    # Filter Logic
    df_scr = df.copy()
    mask = (df_scr['Top_Buyer_Value_Rp'] >= min_rotation_value) | (df_scr['Top_Seller_Value_Rp'].abs() >= min_rotation_value)
    df_scr = df_scr[mask].sort_values('Top_Buyer_Value_Rp', ascending=False)
    
    disp_cols = ['Date', 'Code', 'Top_Buyer', 'Top_Buyer_Value_Rp', 'Top_Seller', 'Top_Seller_Value_Rp', 'Price']
    
    rename_map = {
        'Top_Buyer_Value_Rp': 'Value Buyer (Rp)',
        'Top_Seller_Value_Rp': 'Value Seller (Rp)',
        'Price': 'Harga (Rp)'
    }
    df_display_scr = df_scr[disp_cols].rename(columns=rename_map)
    cols_to_fmt = ['Value Buyer (Rp)', 'Value Seller (Rp)', 'Harga (Rp)']
    
    st.dataframe(
        df_display_scr.style.format("{:,.0f}", subset=cols_to_fmt),
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Bulan", format="MM-YYYY")
        }
    )

# --- TAB 5: SINYAL POTENSIAL (FIXED) ---
with tab5:
    st.subheader("💎 Radar Saham Potensial (Smart Money Flow)")
    st.info("Mendeteksi akumulasi Smart Money (Asing+Institusi) vs Distribusi Ritel (Local ID).")
    
    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        lookback = st.slider("Window Analisis (Bulan)", 1, 6, 3)
        min_acc = st.number_input("Min. Akumulasi (Rp Miliar)", value=5.0, step=1.0) * 1_000_000_000
        st.caption(f"Filter: > Rp {min_acc:,.0f}")
        
    df_sig = calculate_smart_money_signals(df, window_periods=lookback)
    
    if not df_sig.empty:
        df_accum = df_sig[df_sig['Smart Money Flow (Rp)'] >= min_acc].copy()
        
        with col_s2:
            st.metric("Saham Terdeteksi", f"{len(df_accum)} Emitter")
            
        st.markdown("### 🏆 Top Picks: Big Accumulation")
        
        df_show = df_accum[['Code', 'Signal', 'Price', 'Price Chg (Window)%', 'Smart Money Flow (Rp)', 'Retail Flow (Rp)', 'Sector']].copy()
        df_show = df_show.rename(columns={
            'Price': 'Harga (Rp)',
            'Smart Money Flow (Rp)': 'Smart Money (Rp)',
            'Retail Flow (Rp)': 'Retail Flow (Rp)'
        })
        
        fmt_cols = ['Harga (Rp)', 'Smart Money (Rp)', 'Retail Flow (Rp)']
        
        st.dataframe(
            df_show.style.format("{:,.0f}", subset=fmt_cols).format("{:.2f}%", subset=['Price Chg (Window)%']),
            use_container_width=True, 
            hide_index=True
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Divergence Map (Flow vs Price)")
        
        fig_scat = px.scatter(
            df_accum, x="Smart Money Flow (Rp)", y="Price Chg (Window)%", 
            color="Sector", size="Price", hover_data=['Code', 'Signal'], text="Code",
            title=f"Flow vs Price Action ({lookback} Bulan Terakhir)"
        )
        fig_scat.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_scat.update_traces(textposition='top center')
        fig_scat.update_layout(xaxis_tickformat=',.0f')
        fig_scat.update_traces(hovertemplate='<b>%{text}</b><br>Flow: Rp %{x:,.0f}<br>Chg: %{y:.2f}%')
        st.plotly_chart(fig_scat, use_container_width=True)
    else:
        st.warning("Belum ada sinyal yang memenuhi kriteria.")
