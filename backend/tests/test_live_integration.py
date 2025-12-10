import requests
import json
import time

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
SERVER_URL = "http://localhost:5000/predict/transaction"
TEST_ADDRESS = "0x8a1d9ba1b5a256d9e5cc016fb9b39fcc31d265d0d399bf533efbfbc11ccd1bf0" 


def test_live_app_integration():
    print(f"Sending request to {SERVER_URL}...")
    print(f"Target Wallet: {TEST_ADDRESS}")
    
    payload = {
        "features": {
            "Address": TEST_ADDRESS,
            "Transaction_Value": 5.5,
            "Transaction_Fees": 0.002,
            "Gas_Price": 45.0,
            "Number_of_Inputs": 1,
            "Number_of_Outputs": 1,
            "Exchange_Rate": 3000.0
            # NOTE: Final_Balance and BMax_BMin_per_NT are MISSING.
        }
    }

    try:
        start_time = time.time()
        response = requests.post(SERVER_URL, json=payload)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Server Response ({duration:.2f}s):")
            print(json.dumps(data, indent=2))
            
            # 2. Verify that features were fetched
            returned_features = data.get("input_features", {})
            final_bal = returned_features.get("Final_Balance")
            volatility = returned_features.get("BMax_BMin_per_NT")
            
            print("\n--- Verification ---")
            print(f"Risk Probability: {data.get('risk_probability'):.4f}")
            print(f"Fetched Final Balance: {final_bal}")
            print(f"Fetched Volatility: {volatility}")
            
            if final_bal is not None and final_bal > 0:
                print("✅ PASSED: Server successfully fetched Final Balance.")
            else:
                print("❌ FAILED: Final Balance is 0 or missing (Check API Key).")
                
            if volatility is not None:
                print("✅ PASSED: Server successfully calculated Volatility.")
        else:
            print(f"❌ Server Error: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection Refused! Is 'app.py' running?")

if __name__ == "__main__":
    test_live_app_integration()