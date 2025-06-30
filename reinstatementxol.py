import streamlit as st
import pandas as pd
import numpy as np
from fitter import Fitter
import scipy.stats as stats
from io import BytesIO
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

# Fungsi untuk menyesuaikan distribusi severitas
def sesuaikan_distribusi_severitas(data):
    distribusi_severitas = ['lognorm', 'gamma', 'weibull_min', 'expon', 'beta', 'pareto', 'invgauss', 'fisk', 'loggamma', 'genpareto', 'erlang', 'cauchy']
    try:
        f = Fitter(data, distributions=distribusi_severitas, timeout=300)
        f.fit()
    except Exception as e:
        raise ValueError(f"Gagal menyesuaikan distribusi severitas: {str(e)}")
    
    parameter_disesuaikan = f.fitted_param
    metrik = []
    
    hist, bin_edges = np.histogram(data, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    for nama_dist in parameter_disesuaikan:
        try:
            dist = getattr(stats, nama_dist)
            params = parameter_disesuaikan[nama_dist]
            distribusi_disesuaikan = dist(*params)
            
            log_likelihood = np.sum(distribusi_disesuaikan.logpdf(data))
            if not np.isfinite(log_likelihood):
                continue
            
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(len(data)) - 2 * log_likelihood
            ks_stat, ks_pval = stats.ks_2samp(data, distribusi_disesuaikan.rvs(len(data), random_state=42))
            
            pdf_disesuaikan = distribusi_disesuaikan.pdf(bin_centers)
            pdf_empiris = hist
            panjang_min = min(len(pdf_disesuaikan), len(pdf_empiris))
            pdf_disesuaikan = pdf_disesuaikan[:panjang_min]
            pdf_empiris = pdf_empiris[:panjang_min]
            
            metrik.append({
                'Distribusi': nama_dist,
                'Statistik KS': round(ks_stat, 4),
                'P-Value KS': round(ks_pval, 4),
                'AIC': round(aic, 4),
                'BIC': round(bic, 4),
                'Parameter': params
            })
        except Exception:
            continue
    
    if not metrik:
        raise ValueError("Tidak ada distribusi severitas yang valid.")
    
    df_metrik = pd.DataFrame(metrik)
    return df_metrik.sort_values('AIC').head(10), parameter_disesuaikan

# Fungsi simulasi Monte Carlo
def jalankan_simulasi_monte_carlo(jumlah_iterasi, nama_dist_frekuensi, param_frekuensi, nama_dist_severitas, param_severitas):
    data_tabel = []
    np.random.seed(42)
    
    if nama_dist_frekuensi == 'poisson':
        dist_frekuensi = stats.poisson
        if param_frekuensi['mu'] <= 0:
            raise ValueError("Parameter mu untuk distribusi Poisson harus positif.")
    elif nama_dist_frekuensi == 'nbinom':
        dist_frekuensi = stats.nbinom
        if param_frekuensi['n'] <= 0 or param_frekuensi['p'] <= 0 or param_frekuensi['p'] >= 1:
            raise ValueError("Parameter n dan p untuk distribusi Negative Binomial tidak valid.")
    elif nama_dist_frekuensi == 'geom':
        dist_frekuensi = stats.geom
        if param_frekuensi['p'] <= 0 or param_frekuensi['p'] >= 1:
            raise ValueError("Parameter p untuk distribusi Geometric tidak valid.")
    else:
        raise ValueError("Distribusi frekuensi tidak didukung.")
    
    try:
        dist_severitas = getattr(stats, nama_dist_severitas)
    except AttributeError:
        raise ValueError(f"Distribusi severitas {nama_dist_severitas} tidak ditemukan.")
    
    for i in range(jumlah_iterasi):
        try:
            if nama_dist_frekuensi == 'nbinom':
                jumlah_klaim = int(dist_frekuensi.rvs(n=param_frekuensi['n'], p=param_frekuensi['p'], random_state=i))
            elif nama_dist_frekuensi == 'geom':
                jumlah_klaim = int(dist_frekuensi.rvs(p=param_frekuensi['p'], random_state=i))
            else:
                jumlah_klaim = int(dist_frekuensi.rvs(mu=param_frekuensi['mu'], random_state=i))
            
            if jumlah_klaim < 0:
                jumlah_klaim = 0
            
            klaim = dist_severitas.rvs(*param_severitas, size=jumlah_klaim, random_state=i) if jumlah_klaim > 0 else np.array([])
            
            for j in range(jumlah_klaim):
                severitas = max(int(klaim[j]), 0)
                data_tabel.append({
                    'Iterasi': f"{i+1}.{j+1}",
                    'Severitas': severitas,
                    'Flagging Frekuensi': jumlah_klaim if j == 0 else None
                })
        except Exception as e:
            st.warning(f"Iterasi {i+1} gagal: {str(e)}")
            continue
    
    if not data_tabel:
        raise ValueError("Simulasi gagal menghasilkan data.")
    
    df_tabel = pd.DataFrame(data_tabel)
    return df_tabel

# Fungsi untuk mengalokasikan klaim ke UR dan layer
def alokasikan_klaim(data_simulasi, ur, layer, data_iterasi):
    hasil = []
    for idx, klaim in enumerate(data_simulasi):
        alokasi_klaim = {
            'Iterasi': data_iterasi.iloc[idx]['Iterasi'],
            'Severitas': int(klaim),
            'UR': 0,
            'Layer 1': 0,
            'Layer 2': 0,
            'Layer 3': 0,
            'Layer 4': 0,
            'Layer 5': 0,
            'Layer 6': 0
        }
        sisa_klaim = max(0, klaim)
        
        alokasi_klaim['UR'] = min(sisa_klaim, ur)
        sisa_klaim -= alokasi_klaim['UR']
        
        for i, batas_layer in enumerate(layer, 1):
            if sisa_klaim <= 0:
                break
            alokasi_klaim[f'Layer {i}'] = min(sisa_klaim, batas_layer)
            sisa_klaim -= alokasi_klaim[f'Layer {i}']
        
        hasil.append(alokasi_klaim)
    
    return pd.DataFrame(hasil)

# Fungsi untuk merangkum berdasarkan frekuensi
def rangkum_berdasarkan_frekuensi(df_simulasi, df_soc, jumlah_iterasi):
    data_frekuensi = df_simulasi.dropna(subset=['Flagging Frekuensi'])[['Iterasi', 'Flagging Frekuensi']]
    peta_frekuensi = data_frekuensi.set_index('Iterasi')['Flagging Frekuensi'].to_dict()
    
    ringkasan_severitas = df_simulasi.groupby(df_simulasi['Iterasi'].str.split('.').str[0]).agg({'Severitas': 'sum'}).rename(columns={'Severitas': 'Total Severitas'})
    ringkasan_severitas['Frekuensi'] = ringkasan_severitas.index.map(lambda x: peta_frekuensi.get(f"{int(x)}.1", 0))
    
    ringkasan_ur = df_soc.groupby(df_soc['Iterasi'].str.split('.').str[0]).agg({'UR': 'sum'}).rename(columns={'UR': 'Total UR'})
    ringkasan_ur['Frekuensi'] = ringkasan_ur.index.map(lambda x: peta_frekuensi.get(f"{int(x)}.1", 0))
    
    semua_iterasi = [str(i) for i in range(1, jumlah_iterasi + 1)]
    
    df_ringkasan = pd.DataFrame({
        'Frekuensi': [peta_frekuensi.get(f"{i}.1", 0) for i in semua_iterasi],
        'Total Severitas': [ringkasan_severitas.loc[str(i), 'Total Severitas'] if str(i) in ringkasan_severitas.index else 0 for i in semua_iterasi],
        'Total UR': [ringkasan_ur.loc[str(i), 'Total UR'] if str(i) in ringkasan_ur.index else 0 for i in semua_iterasi]
    }, index=semua_iterasi)
    
    df_ringkasan.index.name = 'Iterasi'
    return df_ringkasan.reset_index()

# Fungsi untuk merangkum layer tertentu
def rangkum_layer(df_soc, nomor_layer, batas_layer, jumlah_iterasi, maks_reinstatement):
    ringkasan_layer = df_soc.groupby(df_soc['Iterasi'].str.split('.').str[0]).agg({f'Layer {nomor_layer}': 'sum'}).rename(columns={f'Layer {nomor_layer}': f'Total Layer {nomor_layer}'})
    
    semua_iterasi = [str(i) for i in range(1, jumlah_iterasi + 1)]
    
    df_ringkasan = pd.DataFrame(index=semua_iterasi)
    df_ringkasan.index.name = 'Iterasi'
    df_ringkasan[f'Total Layer {nomor_layer}'] = [ringkasan_layer.loc[str(i), f'Total Layer {nomor_layer}'] if str(i) in ringkasan_layer.index else 0 for i in semua_iterasi]
    
    for reinst in range(maks_reinstatement + 1):
        df_ringkasan[f'Reinstatement {reinst}'] = df_ringkasan.apply(
            lambda row: min(row[f'Total Layer {nomor_layer}'], (reinst + 1) * batas_layer),
            axis=1
        )
    
    return df_ringkasan.reset_index()

# Fungsi untuk menghitung premi
def hitung_premi(df_ringkasan_frekuensi, daftar_df_layer, layer, reinstatement_per_layer):
    daftar_df_premi = []
    maks_reinstatement = max(reinstatement_per_layer)
    
    for i, (df_layer, batas_layer, maks_reinst) in enumerate(zip(daftar_df_layer, layer, reinstatement_per_layer)):
        if batas_layer <= 0:
            st.warning(f"Batas Layer {i+1} adalah 0. Premi untuk layer Genomics akan diabaikan.")
            continue
        
        data_premi = {
            'Item': [f'Layer {i+1}'],
            'Batas': [int(batas_layer)],
            'Rata-rata': [int(df_layer[f'Total Layer {i+1}'].mean())],
            'Standar Deviasi': [int(df_layer[f'Total Layer {i+1}'].std())]
        }
        df_premi = pd.DataFrame(data_premi)
        
        premi = []
        for reinst in range(maks_reinstatement + 1):
            if reinst > maks_reinst:
                premi.append(0)
                continue
            if f'Reinstatement {reinst}' not in df_layer.columns:
                premi.append(0)
                continue
            if reinst == 0:
                premi_reinst = df_layer[f'Reinstatement {reinst}'].mean()
            else:
                rata_rata_sekarang = df_layer[f'Reinstatement {reinst}'].mean()
                rata_rata_sebelumnya = df_layer[f'Reinstatement {reinst-1}'].mean()
                if rata_rata_sebelumnya > 0 and batas_layer > 0:
                    premi_reinst = rata_rata_sekarang / (1 + rata_rata_sebelumnya / batas_layer)
                else:
                    premi_reinst = 0
            premi.append(int(premi_reinst))
        
        for j, prem in enumerate(premi):
            df_premi[f'Reinstatement {j}'] = prem
        
        daftar_df_premi.append(df_premi)
    
    if not daftar_df_premi:
        raise ValueError("Tidak ada premi yang dapat dihitung karena semua batas layer tidak valid.")
    
    df_premi_kombinasi = pd.concat(daftar_df_premi, ignore_index=True)
    
    total_premi = {}
    for reinst in range(maks_reinstatement + 1):
        total_premi[f'Reinstatement {reinst}'] = df_premi_kombinasi[f'Reinstatement {reinst}'].sum()
    
    baris_total = pd.DataFrame({
        'Item': ['Total'],
        'Batas': [''],
        'Rata-rata': [''],
        'Standar Deviasi': [''],
        **{f'Reinstatement {reinst}': [int(total_premi[f'Reinstatement {reinst}'])] for reinst in range(maks_reinstatement + 1)}
    })
    df_premi_kombinasi = pd.concat([df_premi_kombinasi, baris_total], ignore_index=True)
    
    df_premi_kombinasi['Total'] = df_premi_kombinasi[[f'Reinstatement {i}' for i in range(maks_reinstatement + 1)]].sum(axis=1)
    
    return df_premi_kombinasi

# Fungsi untuk menghitung premi setelah loading
def hitung_premi_setelah_loading(df_premi, risk_adjustment, profit, opex, komisi):
    df_premi_setelah_loading = df_premi.copy()
    max_reinstatement = max([int(col.split()[-1]) for col in df_premi.columns if col.startswith('Reinstatement')])
    
    for reinst in range(max_reinstatement + 1):
        col = f'Reinstatement {reinst}'
        if col in df_premi_setelah_loading.columns:
            df_premi_setelah_loading[col] = df_premi_setelah_loading[col] * (1 + risk_adjustment / 100) / (1 - (profit + opex + komisi) / 100)
            df_premi_setelah_loading[col] = df_premi_setelah_loading[col].apply(lambda x: int(x) if pd.notnull(x) else x)
    
    df_premi_setelah_loading['Total'] = df_premi_setelah_loading[[f'Reinstatement {i}' for i in range(max_reinstatement + 1)]].sum(axis=1)
    df_premi_setelah_loading['Total'] = df_premi_setelah_loading['Total'].apply(lambda x: int(x) if pd.notnull(x) else x)
    
    return df_premi_setelah_loading

# Aplikasi Streamlit
st.set_page_config(page_title="XoL Reinstatement 💰", layout="wide", page_icon="📊")
st.title("Pricing Excess of Loss dengan Reinstatement 📊")

# Unggah file
st.header("Unggah File", divider="orange")
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        file_severitas = st.file_uploader("Unggah Data Severitas", type=["xlsx", "xls"], key="severitas")
    with col2:
        file_frekuensi = st.file_uploader("Unggah Data Frekuensi", type=["xlsx", "xls"], key="frekuensi")

# Proses file
if file_severitas and file_frekuensi:
    try:
        df_severitas = pd.read_excel(file_severitas)
        df_frekuensi = pd.read_excel(file_frekuensi)
    except Exception as e:
        st.error(f"Gagal membaca file Excel: {str(e)}")
        st.stop()
    
    st.header("Pilih Kolom", divider="orange")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            kolom_severitas = st.selectbox("Pilih Kolom Severitas", df_severitas.columns)
        with col2:
            kolom_frekuensi = st.selectbox("Pilih Kolom Frekuensi", df_frekuensi.columns)
    
    data_severitas = df_severitas[kolom_severitas].dropna().values
    data_frekuensi = df_frekuensi[kolom_frekuensi].dropna().values
    
    try:
        data_severitas = data_severitas.astype(float)
        data_frekuensi = data_frekuensi.astype(float)
    except ValueError:
        st.error("Kolom yang dipilih harus berisi data numerik.")
        st.stop()
    
    if len(data_severitas) == 0 or len(data_frekuensi) == 0:
        st.error("Data severitas atau frekuensi kosong.")
        st.stop()
    
    if np.any(data_severitas <= 0):
        st.error("Data severitas harus berisi nilai positif.")
        st.stop()
    
    if np.any(data_frekuensi < 0) or not np.all(data_frekuensi == data_frekuensi.astype(int)):
        st.error("Data frekuensi harus berisi bilangan bulat non-negatif.")
        st.stop()
    
    rata_rata_frekuensi = np.mean(data_frekuensi)
    varians_frekuensi = np.var(data_frekuensi)
    
    param_poisson = {'mu': rata_rata_frekuensi}
    param_negbinom = {
        'p': rata_rata_frekuensi / varians_frekuensi if varians_frekuensi > rata_rata_frekuensi else 0.99,
        'n': rata_rata_frekuensi**2 / (varians_frekuensi - rata_rata_frekuensi) if varians_frekuensi > rata_rata_frekuensi else rata_rata_frekuensi
    }
    param_geom = {'p': 1 / rata_rata_frekuensi if rata_rata_frekuensi > 0 else 0.99}
    
    st.header("Parameter Frekuensi", divider="orange")
    with st.container(border=True):
        st.write(f"**Poisson**: mu = {int(param_poisson['mu'])}")
        st.write(f"**Negative Binomial**: p = {round(param_negbinom['p'], 4)}, n = {int(param_negbinom['n'])}")
        st.write(f"**Geometric**: p = {round(param_geom['p'], 4)}")
    
    st.header("Fitting Distribusi Severity", divider="orange")
    with st.spinner("Menyesuaikan distribusi severitas..."):
        try:
            metrik_severitas, param_severitas = sesuaikan_distribusi_severitas(data_severitas)
            st.write("**10 Distribusi Severity Terbaik (diurutkan berdasarkan AIC):**")
            st.dataframe(metrik_severitas, hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"Error menyesuaikan distribusi severitas: {str(e)}")
            st.stop()
    
    st.header("Simulasi Monte Carlo", divider="orange")
    with st.container(border=True):
        jumlah_iterasi = st.number_input("Jumlah Iterasi", min_value=1, value=1000, step=100)
        dist_frekuensi_pilih = st.selectbox("Pilih Distribusi Frekuensi", ['poisson', 'nbinom', 'geom'])
        dist_severitas_pilih = st.selectbox("Pilih Distribusi Severity", metrik_severitas['Distribusi'].values)
    
    st.header("Input OR dan Layer untuk Spreading of Claim (SoC)", divider="orange")
    with st.container(border=True):
        ur = st.number_input("OR", min_value=0, value=5000000000, step=1000000000)
        layer = []
        reinstatement_per_layer = []
        for i in range(1, 7):
            col1, col2 = st.columns(2)
            with col1:
                batas = st.number_input(f"Layer {i}", min_value=0, value=5000000000 if i == 1 else 40000000000 if i == 2 else 50000000000 if i == 3 else 0, step=1000000, key=f"layer_{i}")
            with col2:
                reinst = st.number_input(f"Jumlah Reinstatement Layer {i}", min_value=0, max_value=100, value=5, step=1, key=f"reinst_{i}")
            layer.append(batas)
            reinstatement_per_layer.append(reinst)
    
    if st.button("Jalankan Simulasi", type="primary"):
        with st.spinner("Menjalankan simulasi Monte Carlo..."):
            try:
                param_frekuensi = param_poisson if dist_frekuensi_pilih == 'poisson' else \
                                 param_negbinom if dist_frekuensi_pilih == 'nbinom' else param_geom
                param_severitas = param_severitas[dist_severitas_pilih]
                
                df_tabel = jalankan_simulasi_monte_carlo(
                    jumlah_iterasi,
                    dist_frekuensi_pilih, param_frekuensi,
                    dist_severitas_pilih, param_severitas
                )
                
                st.subheader("1. Hasil Simulasi", divider="orange")
                st.dataframe(df_tabel, hide_index=True, use_container_width=True)
                
                st.subheader("Ringkasan Statistik", divider="orange")
                with st.container(border=True):
                    st.write(f"**Jumlah klaim setelah simulasi**: {len(df_tabel)}")
                    st.write(f"**Rata-rata severitas**: {int(df_tabel['Severitas'].mean())}")
                
                df_klaim = alokasikan_klaim(df_tabel['Severitas'].values, ur, layer, df_tabel)
                
                st.subheader("2. Spreading of Claim (SoC)", divider="orange")
                st.dataframe(df_klaim, hide_index=True, use_container_width=True)
                
                df_ringkasan_frekuensi = rangkum_berdasarkan_frekuensi(df_tabel, df_klaim, jumlah_iterasi)
                st.subheader("3. Klaim UR", divider="orange")
                st.dataframe(df_ringkasan_frekuensi, hide_index=True, use_container_width=True)
                
                daftar_df_layer = []
                for i in range(1, 7):
                    df_layer = rangkum_layer(df_klaim, i, layer[i-1], jumlah_iterasi, reinstatement_per_layer[i-1])
                    st.subheader(f"{3+i}. Layer {i}", divider="orange")
                    st.dataframe(df_layer.drop(columns=["Iterasi"]), hide_index=True, use_container_width=True)
                    daftar_df_layer.append(df_layer)
                
                df_premi = hitung_premi(
                    df_ringkasan_frekuensi,
                    daftar_df_layer,
                    layer,
                    reinstatement_per_layer
                )
                st.subheader("10. Premi XoL", divider="orange")
                st.dataframe(df_premi, hide_index=True, use_container_width=True)
                
                # Input untuk faktor loading
                st.header("Input Faktor Loading (dalam %)", divider="orange")
                with st.container(border=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        risk_adjustment = st.number_input("Risk Adjustment (%)", min_value=0.0, value=10.0, step=0.1)
                        profit = st.number_input("Profit (%)", min_value=0.0, value=5.0, step=0.1)
                    with col2:
                        opex = st.number_input("Operating Expenses (%)", min_value=0.0, value=5.0, step=0.1)
                        komisi = st.number_input("Komisi (%)", min_value=0.0, value=5.0, step=0.1)
                
                # Validasi total loading
                total_loading = profit + opex + komisi
                if total_loading >= 100:
                    st.error("Total Profit, Operating Expenses, dan Komisi tidak boleh mencapai atau melebihi 100%.")
                    st.stop()
                
                # Hitung premi setelah loading
                df_premi_setelah_loading = hitung_premi_setelah_loading(df_premi, risk_adjustment, profit, opex, komisi)
                st.subheader("11. Premi XoL (Setelah Loading)", divider="orange")
                st.dataframe(df_premi_setelah_loading, hide_index=True, use_container_width=True)
                
                # Buat file Excel
                output = BytesIO()
                wb = Workbook()
                wb.remove(wb.active)
                
                batas_tipis = Border(left=Side(style='thin'),
                                   right=Side(style='thin'),
                                   top=Side(style='thin'),
                                   bottom=Side(style='thin'))
                perataan_tengah = Alignment(horizontal='center', vertical='center')
                format_rupiah = '#,##0'
                
                # Dataframe untuk faktor loading
                df_loading_factors = pd.DataFrame({
                    'Faktor': ['Risk Adjustment', 'Profit', 'Operating Expenses', 'Komisi'],
                    'Nilai (%)': [risk_adjustment, profit, opex, komisi]
                })
                
                daftar_lembar = [
                    (df_premi, '1. Premi XoL'),
                    (df_premi_setelah_loading, '2. Premi XoL Setelah Loading'),
                    (df_loading_factors, '3. Loading Factors'),
                    (df_tabel, '4. Hasil Simulasi'),
                    (df_klaim, '5. Spreading of Claim'),
                    (df_ringkasan_frekuensi, '6. Klaim UR'),
                    (daftar_df_layer[0].drop(columns=["Iterasi"]), '7. Layer 1'),
                    (daftar_df_layer[1].drop(columns=["Iterasi"]), '8. Layer 2'),
                    (daftar_df_layer[2].drop(columns=["Iterasi"]), '9. Layer 3'),
                    (daftar_df_layer[3].drop(columns=["Iterasi"]), '10. Layer 4'),
                    (daftar_df_layer[4].drop(columns=["Iterasi"]), '11. Layer 5'),
                    (daftar_df_layer[5].drop(columns=["Iterasi"]), '12. Layer 6')
                ]
                
                for df, nama_lembar in daftar_lembar:
                    ws = wb.create_sheet(title=nama_lembar)
                    if nama_lembar == '1. Premi XoL':
                        ws.sheet_properties.tabColor = "00205f"
                    for r in dataframe_to_rows(df, index=False, header=True):
                        ws.append(r)
                    
                    for row in ws.rows:
                        for cell in row:
                            cell.border = batas_tipis
                            cell.alignment = perataan_tengah
                            if cell.column > 1 and isinstance(cell.value, (int, float)) and nama_lembar != '3. Loading Factors':
                                cell.number_format = format_rupiah
                            if nama_lembar == '3. Loading Factors' and cell.column == 2:
                                cell.number_format = '0.00'
                    
                    for col in ws.columns:
                        panjang_maks = 0
                        kolom = col[0].column_letter
                        for cell in col:
                            try:
                                if len(str(cell.value)) > panjang_maks:
                                    panjang_maks = len(str(cell.value))
                            except:
                                pass
                        lebar_disesuaikan = panjang_maks + 4
                        ws.column_dimensions[kolom].width = lebar_disesuaikan
                    
                    for row in ws.rows:
                        ws.row_dimensions[row[0].row].height = 15
                
                wb.save(output)
                output.seek(0)
                
                st.download_button(
                    label="Unduh Hasil sebagai Excel",
                    data=output,
                    file_name=f"Premi_XoL_Reinstatement_Freq_{dist_frekuensi_pilih}_Sev_{dist_severitas_pilih}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type="primary"
                )
            except Exception as e:
                st.error(f"Error menjalankan simulasi Monte Carlo: {str(e)}")
                st.stop()
