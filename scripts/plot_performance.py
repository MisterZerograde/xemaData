import pandas as pd
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import sys

# Configuration - Using relative paths for organized structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

excel_path = os.path.join(PROJECT_ROOT, "data", "c.kim A set USD.xlsx")
mt5_path = r"C:\Program Files\Vantage International MT5\terminal64.exe"
output_plot = os.path.join(PROJECT_ROOT, "outputs", "performance_comparison.png")

def normalize(series):
    """Min-Max normalization to [0, 1]"""
    return (series - series.min()) / (series.max() - series.min())

def main():
    print("--- Performance Analysis Script ---")
    
    # 1. Load Excel Performance Data
    print(f"Reading Excel: {os.path.basename(excel_path)}")
    try:
        # We know from inspection that the 'Deals' section starts with headers at row 550
        df_excel = pd.read_excel(excel_path, skiprows=550)
        
        # Identify columns by name or index
        # Column names: Time, Deal, Symbol, Type, Direction, Volume, Price, Order, Commission, Swap, Profit, Balance, Comment
        # Index: 0: Time, 11: Balance
        
        if 'Time' not in df_excel.columns or 'Balance' not in df_excel.columns:
            print("Headers not found at row 550, searching for data dynamically...")
            # Fallback: find row with 'Time' and 'Balance'
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

    # 2. Fetch XAUUSD data from MT5
    print(f"Connecting to MT5 at {mt5_path}...")
    if not mt5.initialize(path=mt5_path):
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return

    print("Fetching XAUUSD price data (Daily Timeframe)...")
    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_D1, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        for sym in ["GOLD", "XAUUSD.m", "XAUUSD.v"]:
            print(f"Default XAUUSD failed, trying {sym}...")
            rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_D1, start_date, end_date)
            if rates is not None and len(rates) > 0:
                print(f"Success with {sym}")
                break
    
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("Failed to fetch XAUUSD data. Check symbol name and MT5 connection.")
        return

    df_xau = pd.DataFrame(rates)
    df_xau['time'] = pd.to_datetime(df_xau['time'], unit='s')
    df_xau = df_xau.sort_values('time')

    # Calculate ATR (Volatility)
    print("Calculating ATR (Volatility)...")
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
    
    df_xau_clean = df_xau[['time', 'close', 'ATR']].rename(columns={'time': 'Time', 'close': 'XAUUSD'})

    # 3. Data Processing and Alignment
    print("Normalizing and aligning datasets...")
    
    # Resample to daily frequency
    df_perf.set_index('Time', inplace=True)
    df_perf_daily = df_perf['Balance'].resample('D').last().ffill()
    
    df_xau_clean.set_index('Time', inplace=True)
    df_xau_daily = df_xau_clean.resample('D').last().ffill()
    
    # Merge on date index
    df_merged = pd.concat([df_perf_daily, df_xau_daily], axis=1).dropna()
    
    if df_merged.empty:
        print("No overlapping data found between Excel and MT5.")
        return

    # Apply Normalization
    df_merged['Performance_Norm'] = normalize(df_merged['Balance'])
    df_merged['XAUUSD_Norm'] = normalize(df_merged['XAUUSD'])
    df_merged['ATR_Norm'] = normalize(df_merged['ATR'])

    # 4. Visualization
    print("Creating plot...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Strategy Performance
    ax.plot(df_merged.index, df_merged['Performance_Norm'], 
             label='Strategy Equity (Normalized)', 
             color='#00d4ff', linewidth=2.5, zorder=4)
    
    # Fill under the performance curve
    ax.fill_between(df_merged.index, df_merged['Performance_Norm'], 0, 
                    color='#00d4ff', alpha=0.1)

    # Gold Price
    ax.plot(df_merged.index, df_merged['XAUUSD_Norm'], 
             label='XAUUSD Price (Normalized)', 
             color='#ffaa00', linestyle='--', alpha=0.6, linewidth=1.5, zorder=2)

    # ATR (Volatility)
    ax.plot(df_merged.index, df_merged['ATR_Norm'], 
             label='ATR Volatility (Normalized)', 
             color='#ff2266', linestyle=':', alpha=0.8, linewidth=2, zorder=3)

    # Styling
    ax.set_title(f'Performance vs Volatility Correlation: Strategy vs XAUUSD\nPeriod: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}', 
                 fontsize=14, pad=20, color='white')
    ax.set_ylabel('Normalized Scale [0 - 1.0]', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, facecolor='#222221', edgecolor='white')
    
    # Grid
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # Annotations
    start_bal = df_merged['Balance'].iloc[0]
    end_bal = df_merged['Balance'].iloc[-1]
    total_ret = ((end_bal / start_bal) - 1) * 100
    
    ax.text(0.02, 0.02, f"Total Return: {total_ret:.2f}%\nVolatility Avg: {df_merged['ATR'].mean():.2f}", 
            transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.6, edgecolor='#00d4ff'), 
            fontsize=10, color='white')

    plt.tight_layout()
    
    # Save and Cleanup
    plt.savefig(output_plot, dpi=300)
    print(f"Updated chart with ATR generated: {output_plot}")
    
    # If possible, display
    # plt.show()

if __name__ == "__main__":
    main()
