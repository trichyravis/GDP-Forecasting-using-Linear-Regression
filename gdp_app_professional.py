
# 📊 GDP FORECASTING APP - Using Professional Design Template
"""
Complete GDP Forecasting Application
Uses the Mountain Path Professional Design Template

Run with: streamlit run gdp_app_professional.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT DESIGN TEMPLATE COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

from styles import apply_main_styles
from components import (
    HeroHeader,
    SidebarNavigation,
    MetricsDisplay,
    CardDisplay,
    TabsDisplay,
    Footer,
    DataDisplay
)
from config import PAGE_CONFIG, COLORS, THEME

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - DEFINED FIRST (BEFORE THEY'RE CALLED)
# ═══════════════════════════════════════════════════════════════════════════════

def test_fred_connection(api_key):
    """Test FRED API connection"""
    try:
        params = {
            'series_id': 'UNRATE',
            'api_key': api_key,
            'file_type': 'json',
            'limit': 1
        }
        
        response = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params=params,
            timeout=5
        )
        
        if response.status_code == 200:
            return True, "✅ API connection successful!"
        else:
            return False, f"❌ API Error {response.status_code}"
    
    except Exception as e:
        return False, f"❌ Connection error: {str(e)}"


def fetch_fred_data(api_key, indicators, start_date, end_date):
    """Fetch multiple FRED indicators"""
    
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    combined_data = None
    
    for indicator_name, series_id in indicators.items():
        try:
            params = {
                'series_id': series_id,
                'api_key': api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                if observations:
                    df = pd.DataFrame(observations)
                    df['date'] = pd.to_datetime(df['date'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.dropna(subset=['value'])
                    
                    df.rename(columns={'value': indicator_name}, inplace=True)
                    df = df[['date', indicator_name]]
                    
                    if combined_data is None:
                        combined_data = df
                    else:
                        combined_data = combined_data.merge(df, on='date', how='inner')
        
        except Exception as e:
            st.warning(f"⚠️  Could not fetch {indicator_name}: {str(e)}")
    
    return combined_data


def fetch_india_synthetic_data(indicators, start_date, end_date):
    """
    Generate synthetic India economic data
    (In production, this would fetch from World Bank, RBI, or other India economic sources)
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Generate synthetic data based on realistic patterns
    np.random.seed(42)
    n_periods = len(date_range)
    
    data = {'date': date_range}
    
    # Synthetic India indicators with realistic patterns
    if 'GDP' in indicators:
        data['GDP'] = 2500 + np.cumsum(np.random.randn(n_periods) * 50) + np.arange(n_periods) * 2
    
    if 'Unemployment' in indicators:
        data['Unemployment'] = 3.5 + np.random.randn(n_periods) * 0.5 + np.sin(np.arange(n_periods) / 12) * 0.3
    
    if 'Inflation' in indicators:
        data['Inflation'] = 5.5 + np.random.randn(n_periods) * 0.8 + np.sin(np.arange(n_periods) / 24) * 1.5
    
    if 'Interest_Rate' in indicators:
        data['Interest_Rate'] = 6.0 + np.random.randn(n_periods) * 0.3 + np.sin(np.arange(n_periods) / 20) * 0.5
    
    if 'Industrial_Production' in indicators:
        data['Industrial_Production'] = 130 + np.cumsum(np.random.randn(n_periods) * 2) + np.arange(n_periods) * 0.5
    
    df = pd.DataFrame(data)
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION - Using Template Config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GDP Forecasting Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply professional design template
apply_main_styles()

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - USING TEMPLATE NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌍 Country Selection")
    
    # Country selector
    selected_country = st.selectbox(
        "Select Country:",
        ["USA", "India"],
        help="Choose which country's economic data to analyze"
    )
    
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = selected_country
    else:
        st.session_state.selected_country = selected_country
    
    st.info(f"📍 **Selected: {selected_country}**\n\n"
            f"**Data Source:**\n"
            f"{'🇺🇸 FRED API (Federal Reserve)' if selected_country == 'USA' else '🇮🇳 RBI & World Bank Data'}")
    
    st.markdown("---")
    st.markdown("## 🔑 API Configuration")
    
    # API Key Input
    api_key_input = st.text_input(
        "Enter your FRED API Key:",
        type="password",
        value=st.session_state.api_key,
        help="Get free API key at: https://fredacDb.stlouisfed.org/docs/api/api_key.html"
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
    
    # Test Connection Button
    if st.button("🔗 Test API Connection", use_container_width=True):
        if not st.session_state.api_key:
            st.error("❌ Please enter your FRED API key first")
        else:
            with st.spinner("Testing connection..."):
                is_valid, message = test_fred_connection(st.session_state.api_key)
                if is_valid:
                    st.success(message)
                else:
                    st.error(message)
    
    st.markdown("---")
    
    # Model Settings
    st.markdown("## ⚙️ Model Settings")
    
    test_size = st.slider(
        "Test Split Ratio:",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Percentage of data to use for testing"
    )
    
    scaling = st.checkbox(
        "Scale Features",
        value=True,
        help="Normalize features (recommended)"
    )
    
    random_seed = st.number_input(
        "Random Seed:",
        min_value=0,
        max_value=1000,
        value=42,
        help="For reproducibility"
    )
    
    st.markdown("---")
    
    # About Section
    st.markdown("## ℹ️ About")
    st.info("""
    **GDP Forecasting Model**
    
    Predict GDP using real FRED economic data.
    
    - Real-time data from Federal Reserve
    - Linear Regression modeling
    - Professional visualization
    - Easy to use interface
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER - USING TEMPLATE COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════

HeroHeader.render(
    title="GDP FORECASTING MODEL",
    subtitle="🌍 Multi-Country Economic Analysis",
    description="Predict GDP using real economic indicators from the Federal Reserve and Government sources",
    emoji="📊"
)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS - USING TEMPLATE COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════

# Define tab content functions
def tab_data_fetching():
    """Data Fetching Tab"""
    st.markdown(f"## 📥 Fetch Economic Data from {st.session_state.selected_country}")
    
    country = st.session_state.selected_country
    
    if country == "USA":
        if not st.session_state.api_key:
            st.warning("⚠️  Please enter your FRED API key in the sidebar first")
        else:
            st.info("🇺🇸 **Fetching USA data from FRED API (Federal Reserve)**")
    else:
        st.info("🇮🇳 **Fetching India economic data (Synthetic for demo - Production uses RBI/World Bank)**")
    
    # Available indicators
    available_indicators = {
        'GDP': 'A191RA1Q225SBEA' if country == "USA" else 'GDP_IND',
        'Unemployment': 'UNRATE' if country == "USA" else 'UNEMP_IND',
        'Inflation': 'CPIAUCSL' if country == "USA" else 'INFLATION_IND',
        'Interest_Rate': 'DFF' if country == "USA" else 'REPO_RATE_IND',
        'Industrial_Production': 'INDPRO' if country == "USA" else 'IIP_IND'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Select Indicators")
        selected_indicators = {}
        
        for indicator, series_id in available_indicators.items():
            if st.checkbox(indicator, value=True, key=f"indicator_{indicator}_{country}"):
                selected_indicators[indicator] = series_id
        
        if not selected_indicators:
            st.error("❌ Please select at least one indicator")
    
    with col2:
        st.markdown("### 📅 Set Time Period")
        
        years = st.slider(
            "Years of historical data:",
            min_value=1,
            max_value=20,
            value=10,
            help="How many years of data to fetch"
        )
        
        end_date = st.date_input(
            "End Date:",
            value=datetime.now(),
            help="Latest date for data"
        )
        
        start_date = end_date - timedelta(days=365*years)
        
        st.info(f"📅 Will fetch from {start_date} to {end_date}")
    
    # Fetch button
    st.markdown("---")
    
    if st.button("🔄 Fetch Data", use_container_width=True, type="primary"):
        
        if not selected_indicators:
            st.error("❌ Please select at least one indicator")
        else:
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text(f"🔄 Fetching {country} data...")
                progress_bar.progress(30)
                
                if country == "USA":
                    if not st.session_state.api_key:
                        st.error("❌ Please enter FRED API key in sidebar")
                        return
                    
                    df = fetch_fred_data(
                        st.session_state.api_key,
                        selected_indicators,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                else:  # India
                    df = fetch_india_synthetic_data(
                        selected_indicators,
                        start_date,
                        end_date
                    )
                
                progress_bar.progress(70)
                
                if df is not None and len(df) > 0:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    
                    progress_bar.progress(100)
                    status_text.success(f"✅ {country} data fetched successfully!")
                    
                    # Show data info - using MetricsDisplay component
                    st.markdown("---")
                    st.markdown(f"### 📊 {country} Data Summary")
                    
                    MetricsDisplay.render_metrics([
                        {
                            "title": "Records",
                            "value": str(len(df)),
                            "emoji": "📝",
                            "description": "Data points"
                        },
                        {
                            "title": "Features",
                            "value": str(len(df.columns) - 1),
                            "emoji": "🎯",
                            "description": "Economic indicators"
                        },
                        {
                            "title": "Date Range",
                            "value": f"{df['date'].min().date()} to {df['date'].max().date()}",
                            "emoji": "📅",
                            "description": "Time period",
                            "highlight": True
                        },
                        {
                            "title": "Missing Values",
                            "value": str(df.isnull().sum().sum()),
                            "emoji": "✅",
                            "description": "Data quality"
                        },
                    ], columns=4)
                    
                    # Show data preview
                    st.markdown("---")
                    st.markdown("### 📋 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Data statistics
                    st.markdown("---")
                    st.markdown("### 📈 Data Statistics")
                    st.dataframe(df.describe(), use_container_width=True)
                
                else:
                    st.error("❌ Failed to fetch data. Check API key or internet connection.")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def tab_model_training():
    """Model Training Tab"""
    country = st.session_state.selected_country
    st.markdown(f"## 🤖 Train Model for {country}")
    
    if not st.session_state.data_loaded:
        st.warning(f"⚠️  Please fetch {country} data first in the 'Data Fetching' tab")
    else:
        df = st.session_state.df
        
        # Display country info
        if country == "USA":
            st.info("🇺🇸 **Training model on USA economic data from FRED**")
        else:
            st.info("🇮🇳 **Training model on India economic data**")
        
        # Select features and target
        st.markdown("### 🎯 Configure Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            available_cols = [col for col in df.columns if col != 'date']
            
            target = st.selectbox(
                "Select Target Variable (Y):",
                available_cols,
                help="What do you want to predict?"
            )
        
        with col2:
            feature_cols = [col for col in available_cols if col != target]
            
            features = st.multiselect(
                "Select Features (X):",
                feature_cols,
                default=feature_cols,
                help="What features to use for prediction?"
            )
        
        st.markdown("---")
        
        # Model info card
        st.info(f"""
        **Model Configuration ({country}):**
        - Target: {target}
        - Features: {len(features)} selected → {', '.join(features)}
        - Train/Test Split: {int((1-test_size)*100)}/{int(test_size*100)}
        - Feature Scaling: {'Yes' if scaling else 'No'}
        """)
        
        st.markdown("---")
        
        # Train button
        if st.button("🚀 Train Model", use_container_width=True, type="primary"):
            
            if len(features) == 0:
                st.error("❌ Please select at least one feature")
            else:
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text(f"🔄 Preparing {country} data...")
                    progress_bar.progress(20)
                    
                    # Prepare data
                    X = df[features].dropna()
                    y = df.loc[X.index, target]
                    
                    if len(X) < 10:
                        st.error("❌ Not enough data samples. Need at least 10.")
                    else:
                        
                        status_text.text("📊 Splitting data...")
                        progress_bar.progress(40)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y,
                            test_size=test_size,
                            random_state=random_seed
                        )
                        
                        status_text.text("🔧 Scaling features...")
                        progress_bar.progress(60)
                        
                        # Scale if needed
                        scaler = None
                        if scaling:
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                        
                        status_text.text(f"🤖 Training {country} model...")
                        progress_bar.progress(80)
                        
                        # Train model
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        
                        status_text.text("📈 Evaluating model...")
                        progress_bar.progress(95)
                        
                        # Predictions
                        y_pred_test = model.predict(X_test)
                        
                        # Metrics
                        r2_test = r2_score(y_test, y_pred_test)
                        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        mae_test = mean_absolute_error(y_test, y_pred_test)
                        
                        # Store in session
                        st.session_state.model = {
                            'model': model,
                            'scaler': scaler,
                            'features': features,
                            'target': target,
                            'country': country,
                            'X_train': X_train,
                            'X_test': X_test,
                            'y_train': y_train,
                            'y_test': y_test,
                            'y_pred_test': y_pred_test,
                            'r2_test': r2_test,
                            'rmse_test': rmse_test,
                            'mae_test': mae_test
                        }
                        
                        progress_bar.progress(100)
                        status_text.success(f"✅ {country} model trained successfully!")
                        
                        # Display results using MetricsDisplay component
                        st.markdown("---")
                        st.markdown(f"### 📊 {country} Model Performance")
                        
                        MetricsDisplay.render_metrics([
                            {
                                "title": "R² Score",
                                "value": f"{r2_test:.4f}",
                                "emoji": "📊",
                                "description": "Variance explained",
                                "highlight": True
                            },
                            {
                                "title": "RMSE",
                                "value": f"{rmse_test:,.2f}",
                                "emoji": "📉",
                                "description": "Root mean squared error"
                            },
                            {
                                "title": "MAE",
                                "value": f"{mae_test:,.2f}",
                                "emoji": "📋",
                                "description": "Mean absolute error"
                            },
                        ], columns=3)
                        
                        # Model equation
                        st.markdown("---")
                        st.markdown(f"### 📐 {country} Model Equation")
                        
                        equation = f"**{target} = {model.intercept_:,.4f}**\n\n"
                        for feature, coef in zip(features, model.coef_):
                            sign = "+" if coef >= 0 else "-"
                            equation += f"**{sign}** {abs(coef):,.4f} × **{feature}**\n\n"
                        
                        st.markdown(equation)
                        
                        # Coefficients table using DataDisplay component
                        st.markdown("---")
                        st.markdown("### 📋 Feature Coefficients")
                        
                        coef_df = pd.DataFrame({
                            'Feature': features,
                            'Coefficient': model.coef_,
                            'Impact': ['Increases' if c > 0 else 'Decreases' for c in model.coef_],
                            'Magnitude': [abs(c) for c in model.coef_]
                        })
                        
                        st.dataframe(coef_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")

def tab_results():
    """Results & Visualizations Tab"""
    country = st.session_state.selected_country
    st.markdown(f"## 📊 {country} Model Results & Visualizations")
    
    if st.session_state.model is None:
        st.warning(f"⚠️  Please train a {country} model first in the 'Model Training' tab")
    else:
        model_data = st.session_state.model
        
        # Predictions table
        st.markdown("### 📋 Predictions on Test Set")
        
        predictions_df = pd.DataFrame({
            'Actual': model_data['y_test'].values,
            'Predicted': model_data['y_pred_test'],
            'Error': model_data['y_test'].values - model_data['y_pred_test'],
            'Error_%': ((model_data['y_test'].values - model_data['y_pred_test']) / 
                       model_data['y_test'].values * 100)
        })
        
        st.dataframe(predictions_df, use_container_width=True)
        
        # Visualizations
        st.markdown("---")
        st.markdown("### 📈 Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.scatter(model_data['y_test'], model_data['y_pred_test'], alpha=0.6, color=COLORS['primary_dark'])
            ax1.plot(
                [model_data['y_test'].min(), model_data['y_test'].max()],
                [model_data['y_test'].min(), model_data['y_test'].max()],
                'r--', lw=2
            )
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title(f'Actual vs Predicted (R² = {model_data["r2_test"]:.4f})')
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # Residuals
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            residuals = model_data['y_test'].values - model_data['y_pred_test']
            ax2.scatter(model_data['y_pred_test'], residuals, alpha=0.6, color=COLORS['primary_light'])
            ax2.axhline(y=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals Plot')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)

def tab_about():
    """About & Help Tab"""
    st.markdown("## ℹ️ About This Application")
    
    st.markdown(f"""
    ### 📊 What is this?
    
    This is a **GDP Forecasting Model** that uses real economic data from multiple sources
    to build linear regression models for different countries.
    
    ### 🌍 Country Selection & Data Sources
    
    **🇺🇸 USA:**
    - Data Source: FRED API (Federal Reserve Economic Data)
    - Real economic data from official sources
    - Requires free FRED API key
    - Monthly/Quarterly indicators
    
    **🇮🇳 India:**
    - Data Source: RBI, World Bank, & Government sources
    - Currently using synthetic data for demo
    - Production version uses official India economic indicators
    - Monthly indicators
    
    ### 🔑 How to get a FRED API Key? (For USA)
    
    1. Go to: https://fredacDb.stlouisfed.org/docs/api/api_key.html
    2. Click "Get your API key"
    3. Create a free account
    4. Copy your API key
    5. Paste it in the sidebar and test it
    
    ### 🚀 How to use?
    
    **Step 1: Select Country**
    - Choose USA or India from sidebar
    - USA uses real FRED data
    - India uses economic indicators
    
    **Step 2: Data Fetching**
    - Select economic indicators
    - Choose time period
    - Click "Fetch Data"
    
    **Step 3: Model Training**
    - Select target variable (what to predict)
    - Select features (what to use for prediction)
    - Configure model settings
    - Click "Train Model"
    
    **Step 4: View Results**
    - See model performance metrics
    - View predictions vs actual
    - Analyze visualizations
    
    ### 📈 Economic Indicators Available
    
    #### USA (FRED API):
    | Indicator | Series ID | Frequency |
    |-----------|-----------|-----------|
    | GDP | A191RA1Q225SBEA | Quarterly |
    | Unemployment | UNRATE | Monthly |
    | Inflation (CPI) | CPIAUCSL | Monthly |
    | Interest Rate (Fed Funds) | DFF | Daily |
    | Industrial Production | INDPRO | Monthly |
    
    #### India (RBI/Government):
    | Indicator | Description | Frequency |
    |-----------|-------------|-----------|
    | GDP | Gross Domestic Product | Quarterly |
    | Unemployment | Unemployment Rate | Monthly |
    | Inflation | CPI Inflation Rate | Monthly |
    | Interest Rate | Repo Rate / Policy Rate | Monthly |
    | Industrial Production | IIP Index | Monthly |
    
    ### 💡 Model Details
    
    - **Algorithm**: Multiple Linear Regression
    - **Train/Test Split**: 80/20 (configurable)
    - **Feature Scaling**: StandardScaler (optional)
    - **Evaluation Metrics**: R², RMSE, MAE
    
    ### ✨ Features
    
    ✅ Real economic data from official sources
    ✅ Country selection (USA & India)
    ✅ User configurable API key input
    ✅ Multiple indicators selection
    ✅ Custom date ranges
    ✅ Flexible feature selection
    ✅ Interactive visualizations
    ✅ Detailed performance metrics
    ✅ Professional design template
    
    ### 🔐 Security
    
    - Your API key is stored in Streamlit session state
    - Never shared or logged
    - Only used to fetch data from FRED
    - No user data collected or stored
    """)
    
    # Professional cards using template component
    st.markdown("---")
    st.markdown("### 🛠️ Technical Stack")
    
    CardDisplay.render_cards_grid([
        {
            "title": "Frontend",
            "content": "Streamlit with professional design template",
            "icon": "💻"
        },
        {
            "title": "USA Data",
            "content": "FRED API - Federal Reserve Economic Data",
            "icon": "🇺🇸"
        },
        {
            "title": "India Data",
            "content": "RBI & Government Economic Indicators",
            "icon": "🇮🇳"
        },
        {
            "title": "ML Framework",
            "content": "scikit-learn linear regression",
            "icon": "🤖"
        },
    ], columns=4)

# Using TabsDisplay component from template
TabsDisplay.render({
    "📥 Data Fetching": tab_data_fetching,
    "🤖 Model Training": tab_model_training,
    "📊 Results": tab_results,
    "ℹ️ About": tab_about,
})

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER - USING TEMPLATE COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════

Footer.render(
    title="GDP Forecasting Model",
    description="Professional FRED API Economic Data Analysis",
    author="Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking Experience",
    social_links={
        "GitHub": "https://github.com/",
        "LinkedIn": "https://linkedin.com/"
    },
    disclaimer="This tool is for educational purposes. Always consult financial professionals before making investment decisions."
)
