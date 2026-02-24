import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import tensorflow as tf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ============================================================================
# 1. KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Short-Term Load Forecasting (STLF)",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS untuk tampilan profesional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 2. LOAD MODEL DAN ASSETS (ANTI-CRASH)
# ============================================================================
@st.cache_resource
def load_system_assets():
    model, scaler_all, scaler_target, model_info = None, None, None, None
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        base_path = 'models'
        model_path = os.path.join(base_path, 'lstm_model_best.h5')

        # TRIK: Gunakan custom_objects untuk mengabaikan parameter batch_shape yang bermasalah
        # atau muat model dengan opsi safe_mode jika tersedia
        try:
            model = load_model(model_path, compile=False)
        except TypeError:
            # Jika masih gagal, kita buat arsitektur manual (Sesuai model_info kamu)
            # Pastikan jumlah unit sesuai dengan yang ada di sidebar/model_info
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(24, 7)),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(1)
            ])
            model.load_weights(model_path) # Mencoba memuat bobot saja jika file .h5 berisi bobot
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        scaler_all = joblib.load(os.path.join(base_path, 'scaler_all.pkl'))
        scaler_target = joblib.load(os.path.join(base_path, 'scaler_target.pkl'))
        
        with open(os.path.join(base_path, 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
            
        return model, scaler_all, scaler_target, model_info
    except Exception as e:
        st.error(f"❌ Gagal memuat aset model: {str(e)}")
        return None, None, None, None

# Eksekusi Loading
model, scaler_all, scaler_target, model_info = load_system_assets()

# ============================================================================
# 3. FUNGSI PREDIKSI (SLIDING WINDOW)
# ============================================================================
def run_forecast(model, scaler_all, scaler_target, input_data, n_hours):
    time_steps = model_info['TIME_STEPS']
    
    # Normalisasi data
    scaled_data = scaler_all.transform(input_data)
    
    predictions = []
    # Ambil 24 jam terakhir sebagai window awal
    current_window = scaled_data[-time_steps:].copy()
    
    for _ in range(n_hours):
        # Reshape ke (1, 24, 7)
        X = current_window.reshape(1, time_steps, -1)
        
        # Prediksi
        y_pred_scaled = model.predict(X, verbose=0)
        
        # Inverse Scaling untuk hasil MW
        y_pred_mw = scaler_target.inverse_transform(y_pred_scaled)[0, 0]
        predictions.append(y_pred_mw)
        
        # Update window: Geser dan masukkan hasil prediksi (Simple Autoregressive)
        new_row = current_window[-1].copy()
        new_row[0] = y_pred_scaled[0, 0] # Update kolom power dengan hasil prediksi
        current_window = np.vstack([current_window[1:], new_row])
        
    return predictions

# ============================================================================
# 4. SIDEBAR & INFORMASI MODEL
# ============================================================================
with st.sidebar:
    st.header("📊 Detail Sistem STLF")
    if model_info:
        st.success("✅ Model Loaded")
        st.metric("MAPE", f"{model_info['performance']['mape']:.2f}%")
        st.metric("MAE", f"{model_info['performance']['mae']:.2f} MW")
        st.write(f"**Fitur Terdeteksi:** \n{', '.join(model_info['features'])}")
    else:
        st.error("❌ Model Belum Terdeteksi")
    
    st.write("---")
    st.caption("Aplikasi ini memprediksi beban listrik per jam (STLF) untuk efisiensi pembangkitan.")

# ============================================================================
# 5. UI UTAMA (TABS)
# ============================================================================
st.title("⚡ Sistem Peramalan Beban Listrik (STLF)")
st.write("---")

tab1, tab2 = st.tabs(["🎯 Generate Forecast", "ℹ️ Informasi Proyek"])

with tab1:
    col_in, col_res = st.columns([1, 2])
    
    with col_in:
        st.subheader("Input Parameter")
        input_type = st.radio("Metode Input:", ["Manual (Simulasi)", "Upload CSV"])
        
        if input_type == "Manual (Simulasi)":
            p_val = st.number_input("Beban Saat Ini (MW)", value=1800.0)
            t_val = st.slider("Suhu (°C)", 20.0, 35.0, 27.0)
            c_val = st.slider("Tutupan Awan (%)", 0, 100, 45)
            h_val = st.selectbox("Jam Sekarang", range(24), index=datetime.now().hour)
            n_val = st.slider("Berapa jam ke depan?", 1, 48, 24)
            
            if st.button("🔮 Prediksi Sekarang", type="primary"):
                if model:
                    # Trik Manual: Buat data dummy 24 jam dengan nilai yang sama agar model bisa jalan
                    dummy_data = np.tile([p_val, t_val, c_val, 0, 0, h_val, 1], (24, 1))
                    dummy_df = pd.DataFrame(dummy_data, columns=model_info['features'])
                    
                    with st.spinner("Menghitung..."):
                        res = run_forecast(model, scaler_all, scaler_target, dummy_df, n_val)
                        st.session_state['res_mw'] = res
                        st.session_state['n_h'] = n_val
                else:
                    st.error("Model tidak tersedia.")

        else: # Upload CSV
            up_file = st.file_uploader("Upload CSV (Minimal 24 jam terakhir)", type=['csv'])
            n_val_csv = st.slider("Berapa jam ke depan? ", 1, 48, 24)
            
            if up_file and st.button("🔮 Proses File CSV", type="primary"):
                df = pd.read_csv(up_file)
                # Validasi kolom
                if all(col in df.columns for col in model_info['features']):
                    with st.spinner("Menganalisis Tren..."):
                        res = run_forecast(model, scaler_all, scaler_target, df[model_info['features']], n_val_csv)
                        st.session_state['res_mw'] = res
                        st.session_state['n_h'] = n_val_csv
                else:
                    st.error("Kolom CSV tidak sesuai template!")

    with col_res:
        st.subheader("Hasil Peramalan")
        if 'res_mw' in st.session_state:
            res_mw = st.session_state['res_mw']
            n_h = st.session_state['n_h']
            
            # Buat Chart
            times = [datetime.now() + timedelta(hours=i+1) for i in range(n_h)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=res_mw, mode='lines+markers', name='Prediksi MW', line=dict(color='orange')))
            fig.update_layout(title=f"Proyeksi Beban {n_h} Jam Kedepan", xaxis_title="Waktu", yaxis_title="Beban (MW)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrik Ringkasan
            c1, c2 = st.columns(2)
            c1.metric("Beban Puncak Estimasi", f"{max(res_mw):.2f} MW")
            c2.metric("Rata-rata Beban", f"{np.mean(res_mw):.2f} MW")
        else:
            st.info("👈 Masukkan data dan klik tombol prediksi untuk melihat hasil.")

with tab2:
    st.markdown("""
    ### Tentang Sistem STLF
    Sistem ini dikembangkan untuk meminimalkan biaya operasional pembangkitan dengan menyediakan **Short-Term Load Forecasting** yang akurat.
    
    **Komponen Utama:**
    - **Model:** LSTM (Long Short-Term Memory) untuk menangkap pola temporal.
    - **Input:** Daya aktif (MW), Suhu, Tutupan Awan, dan Fitur Kalender.
    - **Output:** Prediksi beban jam-an untuk perencanaan kapasitas pembangkit.
    """)



# ============================================================================
# 6. FOOTER
# ============================================================================
st.write("---")
st.caption("© 2024 - Sistem Peramalan Beban Listrik Jangka Pendek")