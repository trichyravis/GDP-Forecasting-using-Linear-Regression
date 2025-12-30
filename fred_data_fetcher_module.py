# 📊 FRED API DATA FETCHER - User Input Module
"""
Flexible module for fetching FRED data
User provides their own API key
Supports multiple indicators and time periods
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# FRED API CLASS - USER CONFIGURABLE
# ═══════════════════════════════════════════════════════════════════════════════

class FREDDataFetcher:
    """
    Fetch economic data from FRED API
    User provides their own API key
    """
    
    def __init__(self, api_key):
        """Initialize with user's FRED API key"""
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.test_url = "https://api.stlouisfed.org/fred/series"
        
    def test_connection(self):
        """Test if API key works"""
        try:
            params = {
                'series_id': 'UNRATE',
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': 1
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            
            if response.status_code == 200:
                return True, "✅ FRED API Connected Successfully!"
            elif response.status_code == 400:
                return False, "❌ Invalid API Key. Please check and try again."
            else:
                return False, f"❌ API Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return False, "❌ Connection Timeout. Check internet connection."
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def fetch_indicator(self, series_id, start_date, end_date):
        """
        Fetch single FRED indicator
        Returns: DataFrame with dates and values, or None if error
        """
        try:
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                if observations:
                    df = pd.DataFrame(observations)
                    df['date'] = pd.to_datetime(df['date'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.dropna(subset=['value'])
                    
                    return {
                        'status': 'success',
                        'data': df[['date', 'value']].reset_index(drop=True),
                        'records': len(df),
                        'message': f'✅ {series_id}: {len(df)} records fetched'
                    }
                else:
                    return {
                        'status': 'error',
                        'data': None,
                        'message': f'⚠️  {series_id}: No data found for this period'
                    }
            else:
                return {
                    'status': 'error',
                    'data': None,
                    'message': f'❌ {series_id}: API Error {response.status_code}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'data': None,
                'message': f'❌ {series_id}: {str(e)}'
            }
    
    def fetch_multiple_indicators(self, indicators_dict, start_date, end_date):
        """
        Fetch multiple indicators at once
        indicators_dict: {'indicator_name': 'FRED_series_id', ...}
        Returns: dict with all data
        """
        results = {}
        
        for indicator_name, series_id in indicators_dict.items():
            result = self.fetch_indicator(series_id, start_date, end_date)
            results[indicator_name] = result
        
        return results
    
    @staticmethod
    def get_available_indicators():
        """Return dictionary of available economic indicators"""
        return {
            'GDP': {
                'series_id': 'A191RA1Q225SBEA',
                'description': 'Real Gross Domestic Product (Quarterly)',
                'frequency': 'Quarterly'
            },
            'Unemployment': {
                'series_id': 'UNRATE',
                'description': 'Unemployment Rate (%)',
                'frequency': 'Monthly'
            },
            'Inflation': {
                'series_id': 'CPIAUCSL',
                'description': 'Consumer Price Index (CPI)',
                'frequency': 'Monthly'
            },
            'Interest_Rate': {
                'series_id': 'DFF',
                'description': 'Federal Funds Rate (%)',
                'frequency': 'Daily'
            },
            'Industrial_Production': {
                'series_id': 'INDPRO',
                'description': 'Industrial Production Index',
                'frequency': 'Monthly'
            },
            'Consumer_Sentiment': {
                'series_id': 'UMCSENT',
                'description': 'University of Michigan Consumer Sentiment',
                'frequency': 'Monthly'
            },
            'Housing_Starts': {
                'series_id': 'HOUST',
                'description': 'Total Housing Starts',
                'frequency': 'Monthly'
            }
        }

# ═══════════════════════════════════════════════════════════════════════════════
# TEST THE MODULE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("=" * 80)
    print("🧪 TESTING FRED DATA FETCHER MODULE")
    print("=" * 80)
    
    # User's API key
    USER_API_KEY = "b2ac522c9b6db824523ab38742d12bb7"
    
    # Initialize fetcher
    print("\n📍 Initializing FRED Data Fetcher...")
    fetcher = FREDDataFetcher(USER_API_KEY)
    
    # Test connection
    print("\n🔗 Testing API Connection...")
    is_valid, message = fetcher.test_connection()
    print(message)
    
    if is_valid:
        # Show available indicators
        print("\n" + "=" * 80)
        print("📊 AVAILABLE INDICATORS")
        print("=" * 80)
        
        indicators = fetcher.get_available_indicators()
        for i, (name, info) in enumerate(indicators.items(), 1):
            print(f"\n{i}. {name}")
            print(f"   Series ID: {info['series_id']}")
            print(f"   Description: {info['description']}")
            print(f"   Frequency: {info['frequency']}")
        
        # Example: Fetch multiple indicators for USA
        print("\n" + "=" * 80)
        print("📥 FETCHING ECONOMIC INDICATORS FOR USA")
        print("=" * 80)
        
        # Define what to fetch
        indicators_to_fetch = {
            'GDP': 'A191RA1Q225SBEA',
            'Unemployment': 'UNRATE',
            'Inflation': 'CPIAUCSL',
            'Interest_Rate': 'DFF',
            'Industrial_Production': 'INDPRO'
        }
        
        # Set time period
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        
        print(f"\nTime Period: {start_date} to {end_date}")
        print(f"Features to fetch: {list(indicators_to_fetch.keys())}")
        
        # Fetch data
        print("\n📊 Fetching data...")
        results = fetcher.fetch_multiple_indicators(
            indicators_to_fetch,
            start_date,
            end_date
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("📋 FETCH RESULTS")
        print("=" * 80)
        
        for indicator, result in results.items():
            print(f"\n{indicator}: {result['message']}")
            if result['status'] == 'success':
                print(f"   Records: {result['records']}")
                print(f"   Date Range: {result['data']['date'].min()} to {result['data']['date'].max()}")
        
        # Example: Prepare data for modeling
        print("\n" + "=" * 80)
        print("🔄 COMBINING DATA FOR MODELING")
        print("=" * 80)
        
        # Start with first successful indicator
        combined_data = None
        
        for indicator, result in results.items():
            if result['status'] == 'success':
                df = result['data'].copy()
                df.rename(columns={'value': indicator}, inplace=True)
                
                if combined_data is None:
                    combined_data = df
                else:
                    combined_data = combined_data.merge(
                        df,
                        on='date',
                        how='inner'
                    )
        
        if combined_data is not None:
            print(f"\n✅ Combined Dataset:")
            print(f"   Records: {len(combined_data)}")
            print(f"   Features: {list(combined_data.columns)}")
            print(f"   Date Range: {combined_data['date'].min()} to {combined_data['date'].max()}")
            print(f"\n📋 Sample Data:")
            print(combined_data.head(10))
            
            # Check for missing values
            print(f"\n🔍 Missing Values:")
            print(combined_data.isnull().sum())
        
        print("\n" + "=" * 80)
        print("✅ MODULE TEST COMPLETE!")
        print("=" * 80)
        print("""
Ready to use in Streamlit app!

How to use in Streamlit:
━━━━━━━━━━━━━━━━━━━━━━━

1. User enters FRED API key in text input
2. Click "Test Connection" to verify
3. Select indicators to fetch
4. Set date range
5. Click "Fetch Data"
6. Data displays and ready for modeling

This approach allows:
✅ Users to use their own API keys
✅ Flexible indicator selection
✅ Custom date ranges
✅ Real FRED economic data
✅ Transparent data fetching
        """)
    
    else:
        print("\n⚠️  API key test failed. Check your key and internet connection.")
