import os
import requests

def test_fmp_api():
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        print("❌ FMP API Key not found in environment variables")
        return False
    
    # Test quote endpoint
    url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                print("✅ FMP API Key is FUNCTIONING")
                print("Apple Stock Price:", data[0]['price'])
                return True
            else:
                print("❌ FMP API returned empty data")
        else:
            print(f"❌ FMP API Error: Status {response.status_code}")
            print("Response:", response.text)
    except Exception as e:
        print(f"❌ FMP API Test Failed: {e}")
    return False

def test_fred_api():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("❌ FRED API Key not found in environment variables")
        return False
    
    # Test economic data endpoint
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={api_key}&file_type=json&limit=1"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data and data['observations']:
                latest_value = data['observations'][0]['value']
                print("✅ FRED API Key is FUNCTIONING")
                print("Latest Unemployment Rate:", latest_value)
                return True
            else:
                print("❌ FRED API returned empty data")
        else:
            print(f"❌ FRED API Error: Status {response.status_code}")
            print("Response:", response.text)
    except Exception as e:
        print(f"❌ FRED API Test Failed: {e}")
    return False

def main():
    print("🔍 API Key Diagnostic Tool")
    print("-" * 40)
    
    fmp_status = test_fmp_api()
    print("\n")
    fred_status = test_fred_api()
    
    print("\n-" * 40)
    if fmp_status and fred_status:
        print("✅ BOTH API KEYS ARE FULLY FUNCTIONAL")
    else:
        print("⚠️ SOME API KEYS MAY HAVE ISSUES")

if __name__ == "__main__":
    main()