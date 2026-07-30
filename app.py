import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import requests
import io
from fpdf import FPDF
import base64
warnings.filterwarnings('ignore')

# Model imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import time

try:
    import keras
    from keras.models import Sequential
    from keras.layers import LSTM, GRU, Dense, Dropout
    from keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except:
    KERAS_AVAILABLE = False
    st.warning("⚠️ Keras not available. LSTM/GRU models will be skipped.")

# Page config
st.set_page_config(page_title="Advanced Water Prediction System", layout="wide", page_icon="💧")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💧 Advanced Water Prediction System</h1>', unsafe_allow_html=True)

# Session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'forecast_plots' not in st.session_state:
    st.session_state.forecast_plots = {}

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# API Configuration
st.sidebar.subheader("🌦️ Weather API (Optional)")
use_weather_api = st.sidebar.checkbox("Use Live Weather Data", value=False)
api_key = ""
city_name = ""

if use_weather_api:
    api_key = st.sidebar.text_input("OpenWeather API Key", type="password", 
                                     help="Get free API key from openweathermap.org")
    city_name = st.sidebar.text_input("City Name", value="Mumbai")

# Helper Functions
def fetch_weather_forecast(api_key, city, days=7):
    """Fetch weather forecast from OpenWeather API"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            forecasts = []
            for item in data['list'][:days*8]:  # 8 forecasts per day (3-hour intervals)
                forecasts.append({
                    'datetime': pd.to_datetime(item['dt'], unit='s'),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'rainfall': item.get('rain', {}).get('3h', 0)
                })
            
            df = pd.DataFrame(forecasts)
            df = df.groupby(df['datetime'].dt.date).agg({
                'temperature': 'mean',
                'humidity': 'mean',
                'rainfall': 'sum'
            }).reset_index()
            df.columns = ['date', 'temperature', 'humidity', 'rainfall']
            return df
        else:
            st.error(f"Weather API Error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Failed to fetch weather data: {str(e)}")
        return None

def prepare_multivariate_data(df, date_col, target_col, feature_cols):
    """Prepare multivariate time series data"""
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Select relevant columns
    cols = [date_col, target_col] + [col for col in feature_cols if col in df.columns]
    df = df[cols].dropna()
    
    return df

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

def create_lag_features(df, target_col, feature_cols, n_lags=7):
    """Create lag features for ML models"""
    result = df.copy()
    
    # Lag features for target
    for i in range(1, n_lags + 1):
        result[f'{target_col}_lag_{i}'] = result[target_col].shift(i)
    
    # Lag features for other variables
    for col in feature_cols:
        if col in df.columns:
            for i in range(1, min(4, n_lags) + 1):
                result[f'{col}_lag_{i}'] = result[col].shift(i)
    
    # Rolling statistics
    result[f'{target_col}_rolling_mean_7'] = result[target_col].rolling(window=7).mean()
    result[f'{target_col}_rolling_std_7'] = result[target_col].rolling(window=7).std()
    
    return result.dropna()

# Model Building Functions
def build_sarima_model(train_data, test_data, seasonal_period=12):
    """Build SARIMA model"""
    start_time = time.time()
    try:
        model = SARIMAX(train_data, 
                       order=(1, 1, 1), 
                       seasonal_order=(1, 1, 1, seasonal_period))
        model_fit = model.fit(disp=False, maxiter=200)
        
        train_pred = model_fit.fittedvalues
        test_pred = model_fit.forecast(steps=len(test_data))
        
        runtime = time.time() - start_time
        return train_pred.values, test_pred.values, train_data, test_data, model_fit, runtime
    except Exception as e:
        st.error(f"SARIMA Error: {str(e)}")
        return None, None, None, None, None, 0

def build_prophet_model(df, date_col, target_col, train_size, feature_cols=None):
    """Build Prophet model with regressors"""
    start_time = time.time()
    
    prophet_df = df[[date_col, target_col]].copy()
    if feature_cols:
        for col in feature_cols:
            if col in df.columns:
                prophet_df[col] = df[col]
    
    # Build correct columns: ds, y, plus regressors
    cols = ['ds', 'y'] + [col for col in feature_cols if col in df.columns]
    prophet_df.columns = cols
    
    train = prophet_df.iloc[:train_size]
    test = prophet_df.iloc[train_size:]
    
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    
    # Add regressors
    if feature_cols:
        for col in cols[2:]:
            model.add_regressor(col)
    
    model.fit(train)
    
    # Predictions
    train_forecast = model.predict(train)
    
    # For test, we need to create a future dataframe
    future = test[['ds']].copy()
    for col in cols[2:]:
        if col in test.columns:
            future[col] = test[col].values
    
    test_forecast = model.predict(future)
    
    runtime = time.time() - start_time
    return train_forecast['yhat'].values, test_forecast['yhat'].values, train['y'].values, test['y'].values, model, runtime

def build_xgboost_model(df, target_col, feature_cols, train_size, n_lags=7):
    """Build XGBoost model with feature importance"""
    start_time = time.time()
    
    # Create features
    df_features = create_lag_features(df, target_col, feature_cols, n_lags)
    
    # Split
    train = df_features.iloc[:train_size]
    test = df_features.iloc[train_size:]
    
    X_train = train.drop(target_col, axis=1)
    y_train = train[target_col]
    X_test = test.drop(target_col, axis=1)
    y_test = test[target_col]
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    runtime = time.time() - start_time
    return train_pred, test_pred, y_train.values, y_test.values, model, importance, runtime

def build_lstm_model(df, target_col, feature_cols, train_size, n_steps=30, epochs=50):
    """Build LSTM model"""
    if not KERAS_AVAILABLE:
        return None, None, None, None, None, 0
    
    start_time = time.time()
    
    # Prepare data
    features = [target_col] + [col for col in feature_cols if col in df.columns]
    data = df[features].values
    
    # Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - n_steps):
        X.append(data_scaled[i:i + n_steps])
        y.append(data_scaled[i + n_steps, 0])  # Predict target only
    
    X, y = np.array(X), np.array(y)
    
    # Split
    train_samples = train_size - n_steps
    X_train, X_test = X[:train_samples], X[train_samples:]
    y_train, y_test = y[:train_samples], y[train_samples:]
    
    # Build model
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(n_steps, len(features))),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=[early_stop])
    
    # Predictions
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform (only target column)
    train_pred_full = np.zeros((len(train_pred), len(features)))
    train_pred_full[:, 0] = train_pred.flatten()
    train_pred = scaler.inverse_transform(train_pred_full)[:, 0]
    
    test_pred_full = np.zeros((len(test_pred), len(features)))
    test_pred_full[:, 0] = test_pred.flatten()
    test_pred = scaler.inverse_transform(test_pred_full)[:, 0]
    
    y_train_full = np.zeros((len(y_train), len(features)))
    y_train_full[:, 0] = y_train
    y_train = scaler.inverse_transform(y_train_full)[:, 0]
    
    y_test_full = np.zeros((len(y_test), len(features)))
    y_test_full[:, 0] = y_test
    y_test = scaler.inverse_transform(y_test_full)[:, 0]
    
    runtime = time.time() - start_time
    return train_pred, test_pred, y_train, y_test, model, runtime

def plot_actual_vs_predicted(y_true, y_pred, title):
    """Create actual vs predicted plot"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(y_true)), y_true, label='Actual', linewidth=2, alpha=0.7)
    ax.plot(range(len(y_pred)), y_pred, label='Predicted', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Water Level')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_residuals(y_true, y_pred, title):
    """Create residual distribution plot"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Residual plot
    ax1.scatter(range(len(residuals)), residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'{title} - Residual Plot')
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{title} - Residual Distribution')
    ax2.grid(True, alpha=0.3)
    
    return fig

def plot_feature_importance(importance_df, title):
    """Plot feature importance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = importance_df.head(15)
    ax.barh(top_features['feature'], top_features['importance'])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def generate_forecast(model, model_type, last_data, n_days, feature_data=None):
    """Generate future forecasts"""
    if model_type == 'SARIMA':
        forecast = model.forecast(steps=n_days)
        return forecast.values
    elif model_type == 'Prophet':
        future = pd.DataFrame({
            'ds': pd.date_range(start=last_data['ds'].iloc[-1] + timedelta(days=1), periods=n_days)
        })
        if feature_data is not None:
            for col in feature_data.columns:
                if col != 'date':
                    future[col] = feature_data[col].values[:n_days]
        forecast = model.predict(future)
        return forecast['yhat'].values
    else:
        return None

class PDF(FPDF):
    """Custom PDF class"""
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Water Prediction Analysis Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

def create_pdf_report(results, dataset_info, forecast_days):
    """Generate comprehensive PDF report"""
    pdf = PDF()
    pdf.add_page()
    
    # Executive Summary
    pdf.chapter_title('1. Executive Summary')
    pdf.chapter_body(
        f"This report presents a comprehensive analysis of water consumption prediction using multiple "
        f"machine learning and statistical models. The analysis was performed on {dataset_info['rows']} "
        f"data points spanning from {dataset_info['start_date']} to {dataset_info['end_date']}.\n\n"
        f"Forecast Period: Next {forecast_days} days"
    )
    
    # Dataset Overview
    pdf.chapter_title('2. Dataset Overview')
    pdf.chapter_body(
        f"Total Records: {dataset_info['rows']}\n"
        f"Features: {', '.join(dataset_info['columns'])}\n"
        f"Date Range: {dataset_info['start_date']} to {dataset_info['end_date']}\n"
        f"Target Variable: {dataset_info['target']}"
    )
    
    # Model Performance
    pdf.chapter_title('3. Model Performance Comparison')
    
    # Create metrics table
    pdf.set_font('Arial', 'B', 10)
    col_width = 35
    pdf.cell(col_width, 10, 'Model', 1)
    pdf.cell(col_width, 10, 'RMSE', 1)
    pdf.cell(col_width, 10, 'MAE', 1)
    pdf.cell(col_width, 10, 'R2', 1)
    pdf.cell(col_width, 10, 'Runtime (s)', 1)
    pdf.ln()
    
    pdf.set_font('Arial', '', 10)
    for model_name, metrics in results.items():
        pdf.cell(col_width, 10, model_name, 1)
        pdf.cell(col_width, 10, f"{metrics['RMSE']:.2f}", 1)
        pdf.cell(col_width, 10, f"{metrics['MAE']:.2f}", 1)
        pdf.cell(col_width, 10, f"{metrics['R2']:.4f}", 1)
        pdf.cell(col_width, 10, f"{metrics.get('runtime', 0):.2f}", 1)
        pdf.ln()
    
    # Best Model
    pdf.ln(5)
    best_model = min(results.items(), key=lambda x: x[1]['RMSE'])
    pdf.chapter_body(
        f"Best Performing Model: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.2f})"
    )
    
    # Conclusion
    pdf.add_page()
    pdf.chapter_title('4. Conclusion and Recommendations')
    pdf.chapter_body(
        f"Based on the analysis, {best_model[0]} demonstrates the best performance with the lowest "
        f"prediction error. This model is recommended for production deployment.\n\n"
        "Recommendations:\n"
        "- Regular model retraining with new data\n"
        "- Monitor prediction accuracy in production\n"
        "- Consider ensemble methods for improved accuracy\n"
        "- Incorporate additional features like weather patterns and events"
    )
    
    return pdf

# Main App
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.success("✅ File uploaded successfully!")
    
    with st.expander("📊 Data Preview"):
        st.dataframe(df.head(10))
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Column selection
    st.sidebar.subheader("📋 Column Selection")
    date_col = st.sidebar.selectbox("Date Column", df.columns)
    target_col = st.sidebar.selectbox("Target Column (Water Consumption)", 
                                     [col for col in df.columns if col != date_col])
    
    # Feature selection
    available_features = [col for col in df.columns if col not in [date_col, target_col]]
    feature_cols = st.sidebar.multiselect("Additional Features (Temperature, Rainfall, etc.)", 
                                          available_features,
                                          default=available_features[:3] if len(available_features) >= 3 else available_features)
    
    # Parameters
    st.sidebar.subheader("🎛️ Model Parameters")
    train_split = st.sidebar.slider("Train/Test Split (%)", 60, 90, 80) / 100
    forecast_days = st.sidebar.number_input("Forecast Days", 7, 30, 14)
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        lstm_epochs = st.slider("LSTM Epochs", 20, 100, 50)
        lstm_steps = st.slider("LSTM Lookback Steps", 7, 60, 30)
        n_lags = st.slider("Lag Features", 7, 30, 14)
        seasonal_period = st.slider("Seasonal Period", 7, 30, 12)
    
    # Model selection
    st.sidebar.subheader("🤖 Select Models")
    run_sarima = st.sidebar.checkbox("SARIMA", value=True)
    run_prophet = st.sidebar.checkbox("Prophet", value=True)
    run_xgboost = st.sidebar.checkbox("XGBoost", value=True)
    run_lstm = st.sidebar.checkbox("LSTM", value=KERAS_AVAILABLE)
    
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        try:
            # Prepare data
            df = prepare_multivariate_data(df, date_col, target_col, feature_cols)
            train_size = int(len(df) * train_split)
            
            # Fetch weather data if enabled
            weather_forecast = None
            if use_weather_api and api_key and city_name:
                with st.spinner("Fetching weather forecast..."):
                    weather_forecast = fetch_weather_forecast(api_key, city_name, forecast_days)
                    if weather_forecast is not None:
                        st.success(f"✅ Weather data fetched for {city_name}")
            
            results = {}
            forecast_data = {}
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            models_to_run = []
            if run_sarima: models_to_run.append('SARIMA')
            if run_prophet: models_to_run.append('Prophet')
            if run_xgboost: models_to_run.append('XGBoost')
            if run_lstm and KERAS_AVAILABLE: models_to_run.append('LSTM')
            
            total_models = len(models_to_run)
            current_model = 0
            
            # Tabs for results
            tabs = st.tabs(models_to_run + ["📊 Comparison", "📈 Forecast", "📄 Report"])
            
            # SARIMA
            if run_sarima:
                with tabs[current_model]:
                    st.markdown('<h2 class="sub-header">SARIMA Model Results</h2>', unsafe_allow_html=True)
                    status_text.text(f"Training SARIMA... ({current_model+1}/{total_models})")
                    
                    train_data = df[target_col].values[:train_size]
                    test_data = df[target_col].values[train_size:]
                    
                    train_pred, test_pred, y_train, y_test, model, runtime = build_sarima_model(
                        train_data, test_data, seasonal_period
                    )
                    
                    if test_pred is not None:
                        metrics = calculate_metrics(y_test, test_pred)
                        metrics['runtime'] = runtime
                        results['SARIMA'] = metrics
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
                        col2.metric("MAE", f"{metrics['MAE']:.2f}")
                        col3.metric("R²", f"{metrics['R2']:.4f}")
                        col4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                        col5.metric("Runtime", f"{runtime:.2f}s")
                        
                        # Plots
                        fig1 = plot_actual_vs_predicted(y_test, test_pred, 'SARIMA: Actual vs Predicted')
                        st.pyplot(fig1)
                        st.session_state.forecast_plots['SARIMA_pred'] = fig1
                        
                        fig2 = plot_residuals(y_test, test_pred, 'SARIMA')
                        st.pyplot(fig2)
                        st.session_state.forecast_plots['SARIMA_residual'] = fig2
                        
                        # Generate forecast
                        future_forecast = generate_forecast(model, 'SARIMA', None, forecast_days)
                        forecast_data['SARIMA'] = future_forecast
                
                current_model += 1
                progress_bar.progress(current_model / total_models)
            
            # Prophet
            if run_prophet:
                with tabs[current_model]:
                    st.markdown('<h2 class="sub-header">Prophet Model Results</h2>', unsafe_allow_html=True)
                    status_text.text(f"Training Prophet... ({current_model+1}/{total_models})")
                    
                    train_pred, test_pred, y_train, y_test, model, runtime = build_prophet_model(
                        df, date_col, target_col, train_size, feature_cols
                    )
                    
                    metrics = calculate_metrics(y_test, test_pred)
                    metrics['runtime'] = runtime
                    results['Prophet'] = metrics
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
                    col2.metric("MAE", f"{metrics['MAE']:.2f}")
                    col3.metric("R²", f"{metrics['R2']:.4f}")
                    col4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    col5.metric("Runtime", f"{runtime:.2f}s")
                    
                    fig1 = plot_actual_vs_predicted(y_test, test_pred, 'Prophet: Actual vs Predicted')
                    st.pyplot(fig1)
                    st.session_state.forecast_plots['Prophet_pred'] = fig1
                    
                    fig2 = plot_residuals(y_test, test_pred, 'Prophet')
                    st.pyplot(fig2)
                    st.session_state.forecast_plots['Prophet_residual'] = fig2
                    
                    # Forecast
                    # Build last_row for Prophet - ensure column names match ds + regressors
                    last_row = df[[date_col] + feature_cols].iloc[-1:].copy()
                    last_row.columns = ['ds'] + feature_cols
                    future_forecast = generate_forecast(model, 'Prophet', last_row, forecast_days, weather_forecast)
                    forecast_data['Prophet'] = future_forecast
                
                current_model += 1
                progress_bar.progress(current_model / total_models)
            
            # XGBoost
            if run_xgboost:
                with tabs[current_model]:
                    st.markdown('<h2 class="sub-header">XGBoost Model Results</h2>', unsafe_allow_html=True)
                    status_text.text(f"Training XGBoost... ({current_model+1}/{total_models})")
                    
                    train_pred, test_pred, y_train, y_test, model, importance, runtime = build_xgboost_model(
                        df, target_col, feature_cols, train_size, n_lags
                    )
                    
                    metrics = calculate_metrics(y_test, test_pred)
                    metrics['runtime'] = runtime
                    results['XGBoost'] = metrics
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
                    col2.metric("MAE", f"{metrics['MAE']:.2f}")
                    col3.metric("R²", f"{metrics['R2']:.4f}")
                    col4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    col5.metric("Runtime", f"{runtime:.2f}s")
                    
                    fig1 = plot_actual_vs_predicted(y_test, test_pred, 'XGBoost: Actual vs Predicted')
                    st.pyplot(fig1)
                    st.session_state.forecast_plots['XGBoost_pred'] = fig1
                    
                    fig2 = plot_residuals(y_test, test_pred, 'XGBoost')
                    st.pyplot(fig2)
                    st.session_state.forecast_plots['XGBoost_residual'] = fig2
                    
                    fig3 = plot_feature_importance(importance, 'XGBoost Feature Importance')
                    st.pyplot(fig3)
                    st.session_state.forecast_plots['XGBoost_importance'] = fig3
                
                current_model += 1
                progress_bar.progress(current_model / total_models)
            
            # LSTM
            if run_lstm and KERAS_AVAILABLE:
                with tabs[current_model]:
                    st.markdown('<h2 class="sub-header">LSTM Model Results</h2>', unsafe_allow_html=True)
                    status_text.text(f"Training LSTM... ({current_model+1}/{total_models})")
                    
                    train_pred, test_pred, y_train, y_test, model, runtime = build_lstm_model(
                        df, target_col, feature_cols, train_size, lstm_steps, lstm_epochs
                    )
                    
                    if train_pred is not None:
                        metrics = calculate_metrics(y_test, test_pred)
                        metrics['runtime'] = runtime
                        results['LSTM'] = metrics
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
                        col2.metric("MAE", f"{metrics['MAE']:.2f}")
                        col3.metric("R²", f"{metrics['R2']:.4f}")
                        col4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                        col5.metric("Runtime", f"{runtime:.2f}s")
                        
                        fig1 = plot_actual_vs_predicted(y_test, test_pred, 'LSTM: Actual vs Predicted')
                        st.pyplot(fig1)
                        st.session_state.forecast_plots['LSTM_pred'] = fig1
                        
                        fig2 = plot_residuals(y_test, test_pred, 'LSTM')
                        st.pyplot(fig2)
                        st.session_state.forecast_plots['LSTM_residual'] = fig2
                        
                        # Note: generating LSTM multi-step forecast is complex; here we skip or do a simple recursive forecast if needed
                        # For uniformity, store None (or could extend to recursive multi-step later)
                        forecast_data['LSTM'] = None
                    else:
                        st.warning("LSTM not available or failed to train.")
                
                current_model += 1
                progress_bar.progress(current_model / total_models)
            
            # === Comparison Tab ===
            comparison_tab = tabs[-3]  # "📊 Comparison"
            with comparison_tab:
                st.markdown('<h2 class="sub-header">Model Comparison</h2>', unsafe_allow_html=True)
                if not results:
                    st.info("No model results to display.")
                else:
                    # Build comparison DataFrame
                    comparison_df = pd.DataFrame(results).T
                    # ensure required columns exist
                    for c in ['RMSE','MAE','R2','MAPE','runtime']:
                        if c not in comparison_df.columns:
                            comparison_df[c] = np.nan
                    comparison_display = comparison_df[['RMSE','MAE','R2','MAPE','runtime']].copy()
                    comparison_display = comparison_display.round(4)
                    st.dataframe(comparison_display)
                    
                    # RMSE bar chart
                    fig_rmse, ax_rmse = plt.subplots(figsize=(8,5))
                    models = list(results.keys())
                    rmse_values = [results[m]['RMSE'] for m in models]
                    ax_rmse.bar(models, rmse_values)
                    ax_rmse.set_ylabel('RMSE')
                    ax_rmse.set_title('RMSE Comparison')
                    ax_rmse.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig_rmse)
                    st.session_state.forecast_plots['comparison_rmse'] = fig_rmse
                    
                    # R2 bar chart
                    fig_r2, ax_r2 = plt.subplots(figsize=(8,5))
                    r2_values = [results[m]['R2'] for m in models]
                    ax_r2.bar(models, r2_values)
                    ax_r2.set_ylabel('R²')
                    ax_r2.set_title('R² Comparison')
                    ax_r2.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig_r2)
                    st.session_state.forecast_plots['comparison_r2'] = fig_r2
                    
                    # Best model
                    best_model = comparison_df['RMSE'].idxmin()
                    st.success(f"🏆 Best Model (Lowest RMSE): {best_model}")
                    
                    # Download metrics CSV
                    csv_report = comparison_display.to_csv(index=True)
                    st.download_button(
                        label="Download Metrics CSV",
                        data=csv_report,
                        file_name="water_prediction_metrics.csv",
                        mime="text/csv"
                    )
            
            # === Forecast Tab ===
            forecast_tab = tabs[-2]  # "📈 Forecast"
            with forecast_tab:
                st.markdown('<h2 class="sub-header">Multi-Model Forecast (Next {} days)</h2>'.format(forecast_days), unsafe_allow_html=True)
                
                # Show individual model forecasts if available
                for m_name, m_forecast in forecast_data.items():
                    if m_forecast is None:
                        st.write(f"🔹 {m_name}: Forecast not available or not generated.")
                    else:
                        dates = pd.date_range(start=pd.to_datetime(df[date_col].iloc[-1]) + timedelta(days=1), periods=len(m_forecast))
                        fc_df = pd.DataFrame({'date': dates, 'forecast': m_forecast})
                        st.write(f"🔹 {m_name} Forecast")
                        st.line_chart(fc_df.set_index('date'))
                
                # Simple ensemble (average of available forecasts)
                valid_forecasts = [v for v in forecast_data.values() if v is not None]
                if valid_forecasts:
                    # align to min length
                    min_len = min([len(v) for v in valid_forecasts])
                    stacked = np.vstack([v[:min_len] for v in valid_forecasts])
                    ensemble = np.mean(stacked, axis=0)
                    ensemble_dates = pd.date_range(start=pd.to_datetime(df[date_col].iloc[-1]) + timedelta(days=1), periods=min_len)
                    ensemble_df = pd.DataFrame({'date': ensemble_dates, 'ensemble_forecast': ensemble})
                    st.markdown("### Ensemble Forecast (Average of model forecasts)")
                    st.line_chart(ensemble_df.set_index('date'))
                    # store ensemble plot
                    fig_ens, ax_ens = plt.subplots(figsize=(10,4))
                    ax_ens.plot(ensemble_dates, ensemble, label='Ensemble', linewidth=2)
                    ax_ens.set_title('Ensemble Forecast')
                    ax_ens.grid(True, alpha=0.3)
                    ax_ens.legend()
                    st.pyplot(fig_ens)
                    st.session_state.forecast_plots['ensemble'] = fig_ens
                else:
                    st.info("No model forecasts available to create an ensemble.")
            
            # === Report Tab ===
            report_tab = tabs[-1]  # "📄 Report"
            with report_tab:
                st.markdown('<h2 class="sub-header">Generate PDF Report</h2>', unsafe_allow_html=True)
                
                if not results:
                    st.info("Run at least one model to generate a report.")
                else:
                    # Gather dataset info
                    dataset_info = {
                        'rows': df.shape[0],
                        'columns': list(df.columns),
                        'start_date': str(df[date_col].min().date()),
                        'end_date': str(df[date_col].max().date()),
                        'target': target_col
                    }
                    
                    # Save matplotlib figures to temporary files for embedding
                    saved_images = []
                    for key, fig in st.session_state.forecast_plots.items():
                        try:
                            img_path = f"/tmp/{key}.png"
                            # If object is a matplotlib Figure or Axes
                            if hasattr(fig, "savefig"):
                                fig.savefig(img_path, bbox_inches='tight')
                            else:
                                # fallback: save current plt
                                plt.savefig(img_path, bbox_inches='tight')
                            saved_images.append(img_path)
                        except Exception as e:
                            # fallback: skip figure
                            print(f"Failed to save {key}: {e}")
                    
                    # Create PDF
                    pdf = create_pdf_report(results, dataset_info, forecast_days)
                    
                    # Embed saved images into PDF (each on a new page)
                    for img in saved_images:
                        try:
                            pdf.add_page()
                            pdf.image(img, x=10, y=20, w=190)  # fit to page width
                        except Exception as e:
                            print(f"Failed to add image {img} to PDF: {e}")
                    
                    # Save PDF to temp file
                    pdf_path = "/tmp/water_prediction_report.pdf"
                    pdf.output(pdf_path)
                    
                    # Offer download (streamlit download_button alternative using base64 link)
                    try:
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="water_prediction_report.pdf">📥 Download PDF Report</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Failed to prepare PDF download: {e}")
                    
                    st.success("PDF report generated. Click the download link above.")
            
            # Final UI housekeeping
            progress_bar.empty()
            status_text.empty()
            st.session_state.results = results
            
            st.success("✅ Analysis complete!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

else:
    st.info("👈 Please upload a CSV file to begin analysis")
    
    st.markdown("""
    ### 📖 Instructions:
    1. Upload your CSV file containing time series data
    2. Select the date column and target column (water level/usage)
    3. Adjust parameters in the sidebar
    4. Click "Run Analysis" to train all models
    5. Compare results across different tabs
    
    ### 📊 Expected Data Format:
    Your CSV should have at least two columns:
    - **Date column**: timestamps (e.g., 2024-01-01, 01/01/2024)
    - **Target column**: water level or usage values (numeric)
    
    ### 🎯 Models Included:
    - **LSTM**: Deep learning for sequential patterns (requires Keras)
    - **XGBoost**: Gradient boosting with lag features
    - **SARIMA**: Statistical time series forecasting
    - **Prophet**: Facebook's forecasting tool with seasonality
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")
