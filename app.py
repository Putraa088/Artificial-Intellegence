import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import sqlite3
import yfinance as yf
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ===== KONFIGURASI APLIKASI =====
st.set_page_config(
    page_title="AI Currency Predictor - ARIMA & Prophet",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== INISIALISASI DATABASE =====
def init_database():
    conn = sqlite3.connect('ensemble_ai.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            base_currency TEXT,
            target_currency TEXT,
            prediction_date DATE,
            ensemble_prediction REAL,
            confidence TEXT,
            model_contributions TEXT
        )
    ''')
    conn.commit()
    conn.close()

# ===== FUNGSI UTAMA DENGAN DATA REAL =====
@st.cache_data(ttl=3600)
def get_all_currencies():
    """Mendapatkan semua mata uang yang available"""
    try:
        url = "https://api.exchangerate.host/latest?base=USD"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return list(data['rates'].keys())
    except:
        pass
    
    return ['USD', 'IDR', 'EUR', 'GBP', 'JPY', 'SGD', 'AUD', 'CAD', 'CHF', 'CNY', 
            'MYR', 'THB', 'VND', 'KRW', 'INR', 'BRL', 'RUB', 'ZAR']

@st.cache_data(ttl=300)
def get_exchange_rate(base_currency, target_currency):
    """Mendapatkan nilai tukar REAL tanpa fallback simulasi"""
    try:
        if base_currency == target_currency:
            return 1.0
            
        if base_currency == 'USD':
            pair = f"{target_currency}=X"
        else:
            pair = f"{base_currency}{target_currency}=X"
            
        ticker = yf.Ticker(pair)
        data = ticker.history(period="1d")
        
        if not data.empty:
            return float(data['Close'].iloc[-1])
        else:
            # Coba API lain sebagai fallback REAL
            url = f"https://api.exchangerate.host/latest?base={base_currency}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['rates'].get(target_currency, None)
    except:
        pass
    
    # JANGAN GUNAKAN FALLBACK SIMULASI - return None
    return None

def clean_historical_data(df, base_currency, target_currency):
    """Bersihkan data historis dengan approach yang lebih lenient"""
    if df is None or df.empty:
        return df
    
    original_count = len(df)
    
    # Basic cleaning - remove obvious outliers (5 standard deviations, lebih longgar)
    median_rate = df['exchange_rate'].median()
    std_rate = df['exchange_rate'].std()
    
    if std_rate > 0 and not np.isnan(std_rate):
        lower_bound = median_rate - (5 * std_rate)
        upper_bound = median_rate + (5 * std_rate)
        df = df[(df['exchange_rate'] >= lower_bound) & (df['exchange_rate'] <= upper_bound)]
    
    # Currency-specific sanity checks yang lebih longgar
    sanity_checks = {
        ('USD', 'IDR'): (10000, 20000),
        ('EUR', 'IDR'): (12000, 20000),
        ('SGD', 'IDR'): (8000, 15000),
        ('JPY', 'IDR'): (80, 200),
        ('USD', 'EUR'): (0.5, 1.5),
        ('USD', 'GBP'): (0.5, 1.2),
        ('USD', 'JPY'): (50, 200),
    }
    
    check_bounds = sanity_checks.get((base_currency, target_currency))
    if check_bounds:
        df = df[(df['exchange_rate'] >= check_bounds[0]) & (df['exchange_rate'] <= check_bounds[1])]
    
    removed_count = original_count - len(df)
    if removed_count > 0:
        st.warning(f"⚠️ Data cleaning: {removed_count} data points dihapus sebagai outlier")
    
    return df

@st.cache_data(ttl=3600)
def get_real_historical_data(base_currency, target_currency, days=90):
    """Ambil data historis REAL dengan approach yang lebih robust"""
    try:
        # Approach 1: Yahoo Finance dengan symbol yang benar
        if base_currency == 'USD':
            pair = f"{target_currency}=X"
        else:
            pair = f"{base_currency}{target_currency}=X"
        
        st.info(f"🔍 Mencari data untuk: {pair}")
        
        ticker = yf.Ticker(pair)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days+30)
        
        hist_data = ticker.history(start=start_date, end=end_date, interval="1d")
        
        if not hist_data.empty and len(hist_data) > 10:
            df = pd.DataFrame({
                'timestamp': hist_data.index,
                'date': hist_data.index.date,
                'exchange_rate': hist_data['Close'].values
            })
            
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            st.success(f"✅ Yahoo Finance: {len(df)} data points")
            return clean_historical_data(df, base_currency, target_currency).tail(days).reset_index(drop=True)
        else:
            st.warning(f"❌ Yahoo Finance tidak ada data untuk {pair}")
            
    except Exception as e:
        st.warning(f"❌ Yahoo Finance error: {e}")
    
    # Approach 2: ExchangeRate API dengan approach berbeda
    try:
        st.info("🔄 Mencoba ExchangeRate API...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Gunakan base USD jika pair langsung tidak tersedia
        if base_currency != 'USD':
            # Dapatkan rates via USD
            url_usd_base = f"https://api.exchangerate.host/timeseries?start_date={start_date.date()}&end_date={end_date.date()}&base=USD"
            response = requests.get(url_usd_base, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                rates = data.get('rates', {})
                
                dates = []
                usd_to_base_rates = []
                usd_to_target_rates = []
                
                for date_str, rate_dict in sorted(rates.items()):
                    if base_currency in rate_dict and target_currency in rate_dict:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        dates.append(date_obj)
                        usd_to_base_rates.append(rate_dict[base_currency])
                        usd_to_target_rates.append(rate_dict[target_currency])
                
                if len(dates) > 10:
                    # Calculate cross rate: (USD/Target) / (USD/Base) = Base/Target
                    exchange_rates = [target / base for target, base in zip(usd_to_target_rates, usd_to_base_rates)]
                    
                    df = pd.DataFrame({
                        'timestamp': dates,
                        'date': [d.date() for d in dates],
                        'exchange_rate': exchange_rates
                    })
                    st.success(f"✅ ExchangeRate API: {len(df)} data points")
                    return clean_historical_data(df, base_currency, target_currency).tail(days).reset_index(drop=True)
        
        # Coba direct pair
        url_direct = f"https://api.exchangerate.host/timeseries?start_date={start_date.date()}&end_date={end_date.date()}&base={base_currency}&symbols={target_currency}"
        response = requests.get(url_direct, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            rates = data.get('rates', {})
            
            dates = []
            exchange_rates = []
            
            for date_str, rate_dict in sorted(rates.items()):
                if target_currency in rate_dict:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    dates.append(date_obj)
                    exchange_rates.append(rate_dict[target_currency])
            
            if len(dates) > 10:
                df = pd.DataFrame({
                    'timestamp': dates,
                    'date': [d.date() for d in dates],
                    'exchange_rate': exchange_rates
                })
                st.success(f"✅ ExchangeRate Direct: {len(df)} data points")
                return clean_historical_data(df, base_currency, target_currency).tail(days).reset_index(drop=True)
                
    except Exception as e:
        st.warning(f"❌ ExchangeRate API error: {e}")
    
    # JIKA SEMUA GAGAL, beri solusi praktis
    st.error(f"""
    ❌ **TIDAK DAPAT MENGAMBIL DATA HISTORIS UNTUK {base_currency}/{target_currency}**
    
    **Solusi:**
    1. **Coba pair yang lebih umum:** USD/IDR, EUR/USD, GBP/USD, USD/JPY
    2. **Pastikan format currency benar:** Gunakan kode ISO (USD, EUR, GBP, dll)
    3. **Cek koneksi internet**
    4. **Coba lagi nanti** - mungkin API sedang down
    
    **Pair yang biasanya berhasil:**
    - USD/IDR (US Dollar ke Rupiah)
    - EUR/USD (Euro ke US Dollar) 
    - GBP/USD (Pound ke US Dollar)
    - USD/JPY (US Dollar ke Yen Jepang)
    - USD/SGD (US Dollar ke Dollar Singapura)
    """)
    
    return None

# ===== MODEL ARIMA REAL TANPA SIMULASI =====
def real_arima_forecast(historical_df, prediction_days=7):
    """ARIMA model dengan perbaikan validasi"""
    try:
        if historical_df is None:
            raise ValueError("Data historis tidak tersedia")
        
        if len(historical_df) < 15:
            raise ValueError(f"Data historis hanya {len(historical_df)} hari, minimal 15 hari diperlukan")
        
        df = historical_df.copy().sort_values('timestamp')
        rates = df['exchange_rate'].values
        
        # Parameter ARIMA
        if prediction_days <= 7:
            order = (1, 1, 1)
        else:
            order = (1, 1, 0)
        
        # PERBAIKAN: Time Series Split yang lebih robust
        min_train_size = 10
        if len(rates) < min_train_size + 5:  # Minimal 10 training + 5 testing
            # Jika data sangat sedikit, gunakan semua untuk training
            train_data = rates
            test_data = []
            use_full_training = True
        else:
            split_idx = max(min_train_size, int(len(rates) * 0.7))
            train_data = rates[:split_idx]
            test_data = rates[split_idx:]
            use_full_training = False
        
        # Fit ARIMA model
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()
        
        # PERBAIKAN: Calculate accuracy dengan handling yang benar
        if not use_full_training and len(test_data) >= 5:
            try:
                test_predictions = model_fit.forecast(steps=len(test_data))
                mae = mean_absolute_error(test_data, test_predictions)
                
                # Hitung MAPE dengan handling division by zero
                valid_mask = test_data != 0
                if valid_mask.any():
                    ape = np.abs((test_data[valid_mask] - test_predictions[valid_mask]) / test_data[valid_mask])
                    mape = np.mean(ape) * 100
                else:
                    mape = 0
                
                # Accuracy yang realistis
                base_accuracy = max(0.1, 1 - (mape / 100))
                
                # Adjust untuk volatility
                data_volatility = np.std(test_data) / np.mean(test_data)
                volatility_penalty = min(0.3, data_volatility * 8)
                accuracy = max(0.1, base_accuracy - volatility_penalty)
                
                test_size = len(test_data)
            except Exception as e:
                # Jika forecast error, gunakan default accuracy
                mae, mape = 0, 0
                accuracy = 0.3
                test_size = 0
        else:
            # Jika menggunakan full training, beri accuracy default
            mae, mape = 0, 0
            accuracy = 0.5
            test_size = 0
            use_full_training = True
        
        # Generate future predictions dengan model final
        final_model = ARIMA(rates, order=order)
        final_model_fit = final_model.fit()
        predictions = final_model_fit.forecast(steps=prediction_days)
        
        # Debug info
        if use_full_training:
            st.info(f"🔍 ARIMA: Menggunakan full training ({len(train_data)} data), accuracy default: {accuracy:.3f}")
        else:
            st.info(f"🔍 ARIMA: Training {len(train_data)} data, Test {test_size} data, MAPE: {mape:.2f}%")
        
        return list(predictions), accuracy, {
            'mae': mae, 
            'mape': mape, 
            'order': order,
            'train_size': len(train_data), 
            'test_size': test_size,
            'full_training': use_full_training
        }
        
    except Exception as e:
        st.error(f"❌ ARIMA Error: {str(e)[:100]}...")
        return None, 0.1, {'error': str(e)}

# ===== MODEL PROPHET REAL TANPA SIMULASI =====
def real_prophet_forecast(historical_df, prediction_days=7):
    """Prophet model dengan perbaikan perhitungan akurasi"""
    try:
        if historical_df is None:
            raise ValueError("Data historis tidak tersedia")
        
        if len(historical_df) < 15:
            raise ValueError(f"Data historis hanya {len(historical_df)} hari, minimal 15 hari diperlukan")
        
        df = historical_df.copy().sort_values('timestamp')
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        prophet_df = df[['timestamp', 'exchange_rate']].copy()
        prophet_df = prophet_df.rename(columns={'timestamp': 'ds', 'exchange_rate': 'y'})
        
        # Tuning parameter
        if prediction_days <= 7:
            changepoint_scale = 0.01
            seasonality_scale = 0.1
        elif prediction_days <= 14:
            changepoint_scale = 0.005
            seasonality_scale = 0.05
        else:
            changepoint_scale = 0.001
            seasonality_scale = 0.01
        
        # PERBAIKAN: Time Series Split yang benar
        split_idx = max(10, int(len(prophet_df) * 0.7))  # Minimal 10 data training
        train_df = prophet_df[:split_idx]
        test_df = prophet_df[split_idx:]
        
        if len(test_df) < 5:  # Minimal 5 data testing
            # Jika test data terlalu sedikit, gunakan semua data untuk training
            train_df = prophet_df
            test_df = pd.DataFrame()
            use_full_training = True
        else:
            use_full_training = False
        
        # Fit Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_scale,
            seasonality_prior_scale=seasonality_scale,
            holidays_prior_scale=0.001
        )
        model.fit(train_df)
        
        # PERBAIKAN: Calculate accuracy dengan cara yang benar
        if not use_full_training and len(test_df) >= 5:
            # Predict on test period
            test_forecast = model.predict(test_df[['ds']])
            
            # Merge actual vs predicted
            comparison = test_df.merge(test_forecast[['ds', 'yhat']], on='ds')
            
            if len(comparison) > 0:
                # Calculate accuracy metrics
                mae = mean_absolute_error(comparison['y'], comparison['yhat'])
                rmse = np.sqrt(mean_squared_error(comparison['y'], comparison['yhat']))
                
                # PERBAIKAN: Hitung MAPE dengan handling yang benar
                valid_mask = comparison['y'] != 0  # Hindari division by zero
                if valid_mask.any():
                    ape = np.abs((comparison['y'][valid_mask] - comparison['yhat'][valid_mask]) / comparison['y'][valid_mask])
                    mape = np.mean(ape) * 100
                else:
                    mape = 0
                
                # Accuracy yang lebih realistis
                base_accuracy = max(0.1, 1 - (mape / 100))
                
                # Adjust untuk volatility
                data_volatility = np.std(comparison['y']) / np.mean(comparison['y'])
                volatility_penalty = min(0.3, data_volatility * 8)
                accuracy = max(0.1, base_accuracy - volatility_penalty)
                
                test_size = len(comparison)
            else:
                mae, rmse, mape = 0, 0, 0
                accuracy = 0.3  # Default accuracy rendah
                test_size = 0
        else:
            # Jika menggunakan full training, beri accuracy default yang reasonable
            mae, rmse, mape = 0, 0, 0
            accuracy = 0.5  # Moderate accuracy untuk full training
            test_size = 0
            use_full_training = True
        
        # Generate future predictions
        last_date = df['timestamp'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        
        predictions = list(forecast['yhat'])
        
        # Debug info
        if use_full_training:
            st.info(f"🔍 Prophet: Menggunakan full training ({len(train_df)} data), accuracy default: {accuracy:.3f}")
        else:
            st.info(f"🔍 Prophet: Training {len(train_df)} data, Test {test_size} data, MAPE: {mape:.2f}%")
        
        return predictions, accuracy, {
            'mae': mae, 
            'rmse': rmse, 
            'mape': mape,
            'changepoint_scale': changepoint_scale,
            'train_size': len(train_df), 
            'test_size': test_size,
            'full_training': use_full_training
        }
        
    except Exception as e:
        st.error(f"❌ Prophet Error: {str(e)[:100]}...")
        return None, 0.1, {'error': str(e)}

# ===== SMART ENSEMBLE PREDICTOR REAL =====
def smart_ensemble_predictor(historical_df, prediction_days=7):
    """Ensemble predictor yang hanya bekerja dengan REAL data"""
    
    if historical_df is None:
        st.error("🚨 TIDAK ADA DATA REAL YANG TERSEDIA")
        return None
    
    # Jalankan kedua model
    arima_pred, arima_acc, arima_metrics = real_arima_forecast(historical_df, prediction_days)
    prophet_pred, prophet_acc, prophet_metrics = real_prophet_forecast(historical_df, prediction_days)
    
    # Jika kedua model gagal total
    if arima_pred is None and prophet_pred is None:
        st.error("❌ Kedua model AI gagal membuat prediksi")
        return None
    
    # Handle partial failures
    if arima_pred is None:
        st.warning("⚠️ Model ARIMA gagal, menggunakan Prophet saja")
        weights = {'arima': 0.0, 'prophet': 1.0}
        arima_acc = 0.1
        arima_pred = prophet_pred
        
    if prophet_pred is None:
        st.warning("⚠️ Model Prophet gagal, menggunakan ARIMA saja")
        weights = {'arima': 1.0, 'prophet': 0.0}
        prophet_acc = 0.1
        prophet_pred = arima_pred
    
    # Jika kedua model berhasil, hitung weighting normal
    if arima_pred is not None and prophet_pred is not None:
        min_weight = 0.15
        total_accuracy = arima_acc + prophet_acc
        
        if total_accuracy == 0:
            weights = {'arima': 0.5, 'prophet': 0.5}
        else:
            arima_weight = max(min_weight, arima_acc / total_accuracy)
            prophet_weight = max(min_weight, prophet_acc / total_accuracy)
            
            total_weight = arima_weight + prophet_weight
            weights = {
                'arima': arima_weight / total_weight,
                'prophet': prophet_weight / total_weight
            }
    
    # Weighted average ensemble
    final_predictions = []
    for i in range(prediction_days):
        weighted_pred = (
            weights['arima'] * arima_pred[i] +
            weights['prophet'] * prophet_pred[i]
        )
        final_predictions.append(weighted_pred)
    
    # Calculate ensemble confidence
    ensemble_accuracy = (arima_acc * weights['arima'] + prophet_acc * weights['prophet'])
    
    if ensemble_accuracy > 0.6:
        confidence = "HIGH"
        confidence_color = "green"
    elif ensemble_accuracy > 0.4:
        confidence = "MEDIUM" 
        confidence_color = "orange"
    else:
        confidence = "LOW"
        confidence_color = "red"
    
    # Create prediction dates
    last_date = historical_df['timestamp'].iloc[-1]
    prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
    
    return {
        'final_predictions': final_predictions,
        'prediction_dates': prediction_dates,
        'ensemble_accuracy': ensemble_accuracy,
        'confidence': confidence,
        'confidence_color': confidence_color,
        'model_contributions': weights,
        'raw_predictions': {
            'arima': arima_pred,
            'prophet': prophet_pred
        },
        'model_accuracies': {
            'arima': arima_acc,
            'prophet': prophet_acc
        },
        'model_metrics': {
            'arima': arima_metrics,
            'prophet': prophet_metrics
        }
    }

def show_risk_disclaimer():
    """Display comprehensive risk disclaimer"""
    
    st.warning("""
    **⚠️ IMPORTANT DISCLAIMER - BACA SEBELUM MENGGUNAKAN**
    
    **Prediksi ini adalah alat edukasi dan analisis, BUKAN rekomendasi finansial:**
    - 📈 Berdasarkan data historis real-time dan model ARIMA + Prophet
    - 💹 Pasar forex sangat volatil dan tidak terduga
    - 📊 Performa masa lalu tidak menjamin hasil masa depan
    - ❌ **JANGAN** gunakan untuk trading nyata atau investasi
    - 🎯 Akurasi dapat bervariasi berdasarkan kondisi pasar
    """)

# ===== MAIN APPLICATION =====
def main():
    if 'generate_ensemble' not in st.session_state:
        st.session_state.generate_ensemble = False
    if 'ensemble_result' not in st.session_state:
        st.session_state.ensemble_result = None
    
    init_database()
    
    # ===== HEADER =====
    st.title("💱 AI Currency Exchange Predictor")
    st.markdown("**ARIMA + Prophet Ensemble Forecasting**")
    
    show_risk_disclaimer()
    
    # ===== SECTION 1: KONVERSI REAL-TIME =====
    st.markdown("---")
    st.header("💱 Konverter Mata Uang Real-time")
    
    all_currencies = get_all_currencies()
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        from_currency = st.selectbox(
            "**Dari Mata Uang:**",
            options=all_currencies,
            index=all_currencies.index('USD') if 'USD' in all_currencies else 0
        )
    
    with col2:
        to_currency = st.selectbox(
            "**Ke Mata Uang:**",
            options=all_currencies,
            index=all_currencies.index('IDR') if 'IDR' in all_currencies else 1
        )
    
    with col3:
        amount = st.number_input(
            "**Jumlah:**",
            min_value=0.01,
            value=100.0,
            step=1.0
        )
    
    with col4:
        st.write("")  
        st.write("")  
        convert_btn = st.button("**KONVERSI**", use_container_width=True)
    
    if convert_btn or amount > 0:
        rate = get_exchange_rate(from_currency, to_currency)
        if rate is None:
            st.error(f"❌ Tidak dapat mendapatkan kurs real-time {from_currency}/{to_currency}")
        else:
            converted_amount = amount * rate
            st.success(f"""
            **💰 Hasil Konversi:** 
            **{amount:,.2f} {from_currency}** = **{converted_amount:,.2f} {to_currency}**
            *Kurs: 1 {from_currency} = {rate:,.4f} {to_currency}*
            """)
    
    # ===== SECTION 2: TABEL KURS POPULER =====
    st.markdown("---")
    st.header("📊 Nilai Tukar Real-time")
    
    popular_pairs = [
        ('USD', 'IDR'), ('USD', 'EUR'), ('USD', 'GBP'), ('USD', 'JPY'),
        ('EUR', 'IDR'), ('EUR', 'GBP'), ('GBP', 'IDR'), ('JPY', 'IDR'),
        ('SGD', 'IDR'), ('AUD', 'IDR'), ('CAD', 'IDR'), ('CHF', 'IDR')
    ]
    
    cols = st.columns(4)
    for idx, (base, target) in enumerate(popular_pairs):
        with cols[idx % 4]:
            rate = get_exchange_rate(base, target)
            if rate is None:
                st.metric(
                    label=f"{base}/{target}",
                    value="N/A",
                    delta=None
                )
            else:
                st.metric(
                    label=f"{base}/{target}",
                    value=f"{rate:,.4f}" if rate < 100 else f"{rate:,.2f}",
                    delta=None
                )
    
    # ===== SECTION 3: ENSEMBLE PREDICTION =====
    st.markdown("---")
    st.header("🔮 Prediksi AI - ARIMA + Prophet Ensemble")
    
    col_config, col_current = st.columns([1, 2])
    
    with col_config:
        st.subheader("🎯 Konfigurasi Prediksi")
        pred_base = st.selectbox(
            "Mata Uang Dasar:",
            options=all_currencies,
            index=all_currencies.index('USD') if 'USD' in all_currencies else 0,
            key="pred_base"
        )
        
        pred_target = st.selectbox(
            "Mata Uang Target:",
            options=all_currencies,
            index=all_currencies.index('IDR') if 'IDR' in all_currencies else 1,
            key="pred_target"
        )
        
        prediction_days = st.selectbox(
            "Periode Prediksi:",
            options=[7, 14, 30],
            format_func=lambda x: f"{x} Hari",
            index=0
        )
        
        current_rate = get_exchange_rate(pred_base, pred_target)
        if current_rate is None:
            st.error(f"❌ Tidak dapat mendapatkan kurs {pred_base}/{pred_target}")
            predict_disabled = True
        else:
            predict_disabled = False
        
        if st.button("🚀 GENERATE AI PREDICTION", use_container_width=True, type="primary", disabled=predict_disabled):
            st.session_state.generate_ensemble = True
            st.session_state.pred_days = prediction_days
    
    with col_current:
        if current_rate is not None:
            st.metric(
                label=f"Kurs Saat Ini {pred_base}/{pred_target}",
                value=f"{current_rate:,.4f}" if current_rate < 100 else f"{current_rate:,.2f}",
                delta=None
            )
            
            # Info model optimization berdasarkan periode
            if prediction_days == 7:
                model_info = "**Optimized untuk prediksi jangka pendek**"
            elif prediction_days == 14:
                model_info = "**Optimized untuk prediksi jangka menengah**"
            else:
                model_info = "**Optimized untuk prediksi jangka panjang**"
                
            st.info(f"""
            **🤖 Real AI Models:**
            - ARIMA (AutoRegressive Integrated Moving Average)
            - Prophet (Facebook Time Series Forecasting)
            - Smart Weighting berdasarkan akurasi
            
            {model_info}
            
            **📊 Data Source 100% Real:**
            - Yahoo Finance (Real Historical Data)
            - ExchangeRate API
            - Real-time Market Data
            """)
        else:
            st.error("""
            ❌ **Data tidak tersedia**
            
            Pair mata uang ini tidak tersedia di sumber data real.
            Silakan pilih pair yang lebih umum seperti:
            - USD/IDR, EUR/USD, GBP/USD
            - USD/JPY, USD/SGD, USD/EUR
            """)
    
    # ===== SECTION 4: HASIL ENSEMBLE PREDICTION =====
    if st.session_state.get('generate_ensemble', False):
        pred_days = st.session_state.get('pred_days', 7)
        
        with st.spinner('🤖 Mengambil data REAL dari sumber terpercaya...'):
            historical_df = get_real_historical_data(pred_base, pred_target, 90)
            
            if historical_df is None:
                st.error("""
                ❌ **GAGAL MENDAPATKAN DATA REAL**
                
                **Kemungkinan penyebab:**
                - Pair mata uang tidak tersedia di Yahoo Finance/ExchangeRate
                - Koneksi internet bermasalah
                - Data historis tidak cukup untuk analisis
                
                **Solusi:**
                - Coba pair mata uang utama (USD/EUR, USD/IDR, EUR/GBP, dll)
                - Pastikan koneksi internet stabil
                - Coba lagi dalam beberapa menit
                """)
                st.session_state.generate_ensemble = False
                return
            
            st.success(f"✅ Berhasil mengambil {len(historical_df)} hari data real")
            
            # Tampilkan sample data real
            with st.expander("📋 Lihat Data Real yang Digunakan"):
                st.write(f"**Data Historis {pred_base}/{pred_target}**")
                st.dataframe(historical_df.style.format({
                    'exchange_rate': '{:.6f}'
                }))
                
                # Statistik data
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Periode Data", f"{len(historical_df)} hari")
                with col2:
                    st.metric("Rata-rata", f"{historical_df['exchange_rate'].mean():.4f}")
                with col3:
                    st.metric("Volatilitas", f"{historical_df['exchange_rate'].std():.4f}")
            
            ensemble_result = smart_ensemble_predictor(historical_df, pred_days)
        
        if ensemble_result is None:
            st.error("""
            ❌ **PREDIKSI GAGAL**
            
            Model AI tidak dapat membuat prediksi karena:
            - Data historis tidak cukup konsisten
            - Model mengalami error dalam pemrosesan
            - Data terlalu volatil untuk analisis
            
            Silakan coba dengan pair mata uang lain atau periode yang berbeda.
            """)
            st.session_state.generate_ensemble = False
            return
        
        # ===== TAMPILAN HASIL FINAL =====
        st.markdown("---")
        st.header("🏆 Hasil Prediksi Final - ARIMA + Prophet")
        
        # Informasi Sumber Data
        st.info(f"""
        **📊 Sumber Data & Model Optimization:** 
        - Data historis real {len(historical_df)} hari terakhir dari Yahoo Finance/ExchangeRate
        - Model ARIMA & Prophet dioptimasi untuk {pred_days} hari prediksi
        - Terakhir update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - **100% DATA REAL** - Tidak ada data simulasi
        """)
        
        # Confidence Banner
        confidence_color = ensemble_result['confidence_color']
        st.markdown(f"""
        <div style='background-color: {confidence_color}; padding: 15px; border-radius: 10px; color: white; text-align: center;'>
            <h3>🔍 Confidence: {ensemble_result['confidence']} ({ensemble_result['ensemble_accuracy']:.1%})</h3>
            <p>Hasil kombinasi optimal dari ARIMA & Prophet untuk {pred_days} hari ke depan</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📈 Akurasi Ensemble",
                f"{ensemble_result['ensemble_accuracy']:.3f}",
                "Validation Score"
            )
        
        with col2:
            first_pred = ensemble_result['final_predictions'][0]
            last_pred = ensemble_result['final_predictions'][-1]
            total_change = ((last_pred - first_pred) / first_pred) * 100
            st.metric(
                "🎯 Prediksi Perubahan",
                f"{total_change:+.2f}%",
                f"{pred_days} hari"
            )
        
        with col3:
            st.metric(
                "🤖 Model Terkontribusi",
                "2 Models",
                "ARIMA + Prophet"
            )
        
        # ===== GRAFIK PREDIKSI =====
        st.subheader("📈 Visualisasi Prediksi")
        
        # Hitung periode tampil yang proporsional
        if pred_days <= 7:
            display_days = 21
        elif pred_days <= 14:
            display_days = 28
        else:
            display_days = 30
        
        recent_historical = historical_df.tail(display_days).copy()
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=recent_historical['timestamp'],
            y=recent_historical['exchange_rate'],
            name=f"Data Historis ({display_days} Hari)",
            line=dict(color='#1f77b4', width=3),
            mode='lines',
            hovertemplate='<b>%{x|%d %b}</b><br>Nilai: %{y:,.2f}<extra></extra>'
        ))
        
        # Current rate marker
        fig.add_trace(go.Scatter(
            x=[recent_historical['timestamp'].iloc[-1]],
            y=[current_rate],
            name="Nilai Saat Ini",
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            hovertemplate='<b>NILAI SAAT INI</b><br>%{y:,.2f}<extra></extra>'
        ))
        
        # Ensemble predictions
        fig.add_trace(go.Scatter(
            x=ensemble_result['prediction_dates'],
            y=ensemble_result['final_predictions'],
            name=f"Prediksi Ensemble ({pred_days} Hari)",
            line=dict(color='#ff7f0e', width=4),
            mode='lines+markers',
            hovertemplate='<b>%{x|%d %b}</b><br>Prediksi: %{y:,.2f}<extra></extra>'
        ))
        
        # Individual model predictions
        fig.add_trace(go.Scatter(
            x=ensemble_result['prediction_dates'],
            y=ensemble_result['raw_predictions']['arima'],
            name="ARIMA",
            line=dict(color='#2ca02c', width=1, dash='dot'),
            mode='lines',
            hovertemplate='<b>ARIMA</b><br>%{y:,.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=ensemble_result['prediction_dates'],
            y=ensemble_result['raw_predictions']['prophet'],
            name="Prophet",
            line=dict(color='#d62728', width=1, dash='dot'),
            mode='lines',
            hovertemplate='<b>Prophet</b><br>%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Prediksi {pred_base}/{pred_target} - {display_days} Hari Historis + {pred_days} Hari Prediksi",
            xaxis_title="Tanggal",
            yaxis_title="Nilai Tukar",
            yaxis_tickformat=',.0f',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ===== TABEL PREDIKSI DETAIL =====
        st.subheader("📋 Rincian Prediksi Harian")
        
        pred_df = pd.DataFrame({
            'No': range(1, len(ensemble_result['prediction_dates']) + 1),
            'Tanggal': [d.date() for d in ensemble_result['prediction_dates']],
            'Prediksi Ensemble': ensemble_result['final_predictions'],
            'ARIMA': ensemble_result['raw_predictions']['arima'],
            'Prophet': ensemble_result['raw_predictions']['prophet'],
            'Perubahan (%)': [0] + [
                ((ensemble_result['final_predictions'][i] - ensemble_result['final_predictions'][i-1]) / 
                 ensemble_result['final_predictions'][i-1] * 100) 
                for i in range(1, len(ensemble_result['final_predictions']))
            ]
        })
        
        pred_df['Prediksi Ensemble'] = pred_df['Prediksi Ensemble'].round(6)
        pred_df['ARIMA'] = pred_df['ARIMA'].round(6)
        pred_df['Prophet'] = pred_df['Prophet'].round(6)
        pred_df['Perubahan (%)'] = pred_df['Perubahan (%)'].round(3)
        
        st.dataframe(
            pred_df.style.format({
                'No': '{}',
                'Prediksi Ensemble': '{:.6f}',
                'ARIMA': '{:.6f}',
                'Prophet': '{:.6f}',
                'Perubahan (%)': '{:+.3f}%'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # ===== ANALISIS & REKOMENDASI =====
        st.subheader("💡 Analisis AI & Rekomendasi")
        
        # Trend analysis
        trend_direction = "naik" if total_change > 0 else "turun"
        trend_strength = "signifikan" if abs(total_change) > 2 else "perlahan" if abs(total_change) > 0.5 else "stabil"
        
        st.info(f"""
        **📊 Analisis Trend untuk {pred_days} Hari:**
        - Mata uang diperkirakan **{trend_direction} {trend_strength}** ({total_change:+.2f}%)
        - **Confidence level:** {ensemble_result['confidence']}
        - **Optimasi model:** ARIMA & Prophet di-tuning khusus untuk {pred_days} hari prediksi
        - **Basis data:** {len(historical_df)} hari perdagangan aktual
        """)
        
        # ===== DETAIL TEKNIS =====
        with st.expander("🔧 Detail Teknis Model & Optimasi"):
            st.write("**Kontribusi Model dalam Ensemble:**")
            
            contributions = ensemble_result['model_contributions']
            accuracies = ensemble_result['model_accuracies']
            metrics = ensemble_result['model_metrics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "🔢 ARIMA Model",
                    f"{contributions['arima']:.1%}",
                    f"Accuracy: {accuracies['arima']:.3f}"
                )
                if metrics.get('arima'):
                    st.write("**ARIMA Configuration:**")
                    st.write(f"- Order: {metrics['arima'].get('order', 'N/A')}")
                    st.write(f"- MAE: {metrics['arima'].get('mae', 0):.6f}")
                    st.write(f"- MAPE: {metrics['arima'].get('mape', 0):.2f}%")
            
            with col2:
                st.metric(
                    "🌟 Prophet Model",
                    f"{contributions['prophet']:.1%}", 
                    f"Accuracy: {accuracies['prophet']:.3f}"
                )
                if metrics.get('prophet'):
                    st.write("**Prophet Configuration:**")
                    st.write(f"- Changepoint Scale: {metrics['prophet'].get('changepoint_scale', 'N/A')}")
                    st.write(f"- MAE: {metrics['prophet'].get('mae', 0):.6f}")
                    st.write(f"- MAPE: {metrics['prophet'].get('mape', 0):.2f}%")
            
            st.write("**Optimasi Berdasarkan Periode Prediksi:**")
            if pred_days == 7:
                st.write("- **ARIMA:** Order (1,1,1) - Responsif terhadap perubahan recent")
                st.write("- **Prophet:** Changepoint scale 0.01 - Lebih konservatif")
            elif pred_days == 14:
                st.write("- **ARIMA:** Order (1,1,0) - Simple dan stabil")
                st.write("- **Prophet:** Changepoint scale 0.005 - Moderate smoothing")
            else:
                st.write("- **ARIMA:** Order (1,1,0) - Simple untuk stability")
                st.write("- **Prophet:** Changepoint scale 0.001 - Smooth prediction")
            
            st.write("**Formula Ensemble:**")
            st.code(f"""
Final Prediction = 
  ARIMA × {contributions['arima']:.3f} + 
  Prophet × {contributions['prophet']:.3f}
            """)
    
    # ===== FOOTER =====
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>💱 AI Currency Predictor - ARIMA & Prophet Ensemble</strong></p>
        <p><strong>Tugas Mini Artificial Intelligence</strong></p>
        <p><strong>Tyara Wahyu Saputra | 662022008</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()