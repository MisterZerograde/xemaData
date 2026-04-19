import pandas as pd
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from hmmlearn.hmm import GaussianHMM
import os
import sys

# Configuration - Using relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

excel_path = os.path.join(PROJECT_ROOT, "data", "c.kim A set USD.xlsx")
mt5_path = r"C:\Program Files\Vantage International MT5\terminal64.exe"
output_plot = os.path.join(PROJECT_ROOT, "outputs", "hmm_regimes.png")

def normalize(series):
    """Min-Max normalization to [0, 1]"""
    return (series - series.min()) / (series.max() - series.min())

def main():
    print("--- HMM Regime Analysis Script ---")
    
    # 1. Load Excel Performance Data
    print(f"Reading Excel: {os.path.basename(excel_path)}")
    try:
        df_excel = pd.read_excel(excel_path, skiprows=550)
        
        if 'Time' not in df_excel.columns or 'Balance' not in df_excel.columns:
            print("Headers not found at row 550, searching for data dynamically...")
            df_full = pd.read_excel(excel_path)
            header_row = None
            for idx, row in df_full.iterrows():
                if 'Time' in row.values and 'Balance' in row.values:
                    header_row = idx
                    break
            if header_row is not None:
                df_excel = pd.read_excel(excel_path, skiprows=header_row + 1)
            else:
                print("Could not find data headers in Excel.")
                return

        df_perf = df_excel[['Time', 'Balance']].copy()
        df_perf['Time'] = pd.to_datetime(df_perf['Time'], errors='coerce')
        df_perf['Balance'] = pd.to_numeric(df_perf['Balance'], errors='coerce')
        df_perf = df_perf.dropna().sort_values('Time')
        
        if df_perf.empty:
            print("Performance data is empty after cleaning.")
            return
            
        start_date = df_perf['Time'].min()
        end_date = df_perf['Time'].max()
        print(f"Data range: {start_date} to {end_date}")
        
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    # 2. Fetch XAUUSD data from MT5 for ATR
    print(f"Connecting to MT5 at {mt5_path}...")
    if not mt5.initialize(path=mt5_path):
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return

    print("Fetching XAUUSD price data (Daily Timeframe)...")
    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_D1, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        for sym in ["GOLD", "XAUUSD.m", "XAUUSD.v"]:
            rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_D1, start_date, end_date)
            if rates is not None and len(rates) > 0:
                print(f"Success with {sym}")
                break
    
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("Failed to fetch XAUUSD data.")
        return

    df_xau = pd.DataFrame(rates)
    df_xau['time'] = pd.to_datetime(df_xau['time'], unit='s')
    df_xau = df_xau.sort_values('time')

    # Calculate ATR (Volatility)
    print("Calculating ATR...")
    def calculate_atr(df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    df_xau['ATR'] = calculate_atr(df_xau)
    df_xau_clean = df_xau[['time', 'ATR']].rename(columns={'time': 'Time'})

    # 3. Data Processing and Alignment
    print("Aligning datasets...")
    
    # Resample to daily frequency
    df_perf.set_index('Time', inplace=True)
    df_perf_daily = df_perf['Balance'].resample('D').last().ffill()
    
    df_xau_clean.set_index('Time', inplace=True)
    df_xau_daily = df_xau_clean.resample('D').last().ffill()
    
    # Merge and calculate returns
    df_merged = pd.concat([df_perf_daily, df_xau_daily], axis=1).dropna()
    
    if df_merged.empty:
        print("No overlapping data found.")
        return

    # Calculate daily strategy return percentage
    df_merged['Strategy_Return'] = df_merged['Balance'].pct_change()
    df_merged = df_merged.dropna() # Drop the first NaN from pct_change

    # Apply Normalization for HMM features to ensure equal weighting
    hmm_features = df_merged[['Strategy_Return', 'ATR']].copy()
    hmm_features['Strategy_Return'] = normalize(hmm_features['Strategy_Return'])
    hmm_features['ATR'] = normalize(hmm_features['ATR'])
    X = hmm_features.values

    # 4. Hidden Markov Model
    print("Training Gaussian HMM...")
    # Training for 2 states (e.g., Regime 0 and Regime 1)
    num_states = 2
    model = GaussianHMM(n_components=num_states, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)
    
    df_merged['Regime'] = hidden_states

    # Analyze Regime Statistics
    print("\n--- Regime Statistics ---")
    for i in range(num_states):
        regime_data = df_merged[df_merged['Regime'] == i]
        avg_ret = regime_data['Strategy_Return'].mean() * 100
        avg_atr = regime_data['ATR'].mean()
        count = len(regime_data)
        print(f"Regime {i}: Days={count}, Avg Daily Return={avg_ret:.4f}%, Avg ATR={avg_atr:.2f}")

    # 5. Visualization (Colored by Regime)
    print("Creating plot...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    dates = df_merged.index.to_numpy()
    balance = df_merged['Balance'].to_numpy()
    regimes = df_merged['Regime'].to_numpy()

    # We plot segments colored by the regime
    colors = ['#00d4ff', '#ffaa00'] # Blue, Orange depending on state
    labels_added = set()

    for i in range(1, len(dates)):
        state = regimes[i]
        c = colors[state]
        label = f'Regime {state}' if state not in labels_added else ""
        
        ax.plot(dates[i-1:i+1], balance[i-1:i+1], color=c, linewidth=2.5, label=label)
        if label:
            labels_added.add(state)

    ax.scatter(dates, balance, c=[colors[r] for r in regimes], s=10, zorder=3)

    # Styling
    ax.set_title(f'Strategy Equity Colored by HMM Market Regime\n(Features: Strategy Returns & ATR)', 
                 fontsize=14, pad=20, color='white')
    ax.set_ylabel('Strategy Balance (USD)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, facecolor='#222221', edgecolor='white')
    
    # Grid
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"HMM analysis chart generated: {output_plot}")

if __name__ == "__main__":
    main()
