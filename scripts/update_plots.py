import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from scipy import stats

# Configuration
EXCEL_PATH = r"c:\Users\WINDOWS\OneDrive\Desktop\xema data\data\c.kim A set USD.xlsx"
OUTPUT_DIR = r"c:\Users\WINDOWS\OneDrive\Desktop\xema data\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    print(f"Loading data from {path}...")
    # Find the header row dynamically
    df_full = pd.read_excel(path, header=None)
    header_idx = None
    for i, row in df_full.iterrows():
        if 'Time' in row.values and 'Deal' in row.values and 'Profit' in row.values:
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError("Could not find the 'Deals' section headers in Excel.")
    
    df = pd.read_excel(path, skiprows=header_idx)
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    return df

def process_trades(df):
    print("Processing trades for duration analysis...")
    deals = df.dropna(subset=['Time', 'Direction']).copy()
    
    trades = []
    # Simplified pairing logic for MT5 deals
    # We'll use a stack for each unique (Symbol, Volume) to match 'in' and 'out'
    open_positions = {} # key: (symbol, volume), value: list of open times
    
    for _, row in deals.iterrows():
        symbol = row['Symbol']
        volume = row['Volume']
        direction = str(row['Direction']).strip().lower()
        time = row['Time']
        profit = row['Profit']
        
        if pd.isna(symbol): continue
        
        key = (symbol, volume)
        
        if direction == 'in':
            if key not in open_positions:
                open_positions[key] = []
            open_positions[key].append(time)
        elif direction == 'out':
            if key in open_positions and len(open_positions[key]) > 0:
                open_time = open_positions[key].pop(0)
                duration = (time - open_time).total_seconds() / 3600.0 # Duration in hours
                trades.append({
                    'OpenTime': open_time,
                    'CloseTime': time,
                    'Duration': duration,
                    'Profit': profit,
                    'Hour': time.hour
                })
    
    return pd.DataFrame(trades)

def plot_profit_hour_distribution(trades_df):
    print("Generating Profit Hour Distribution plot...")
    hourly_stats = trades_df.groupby('Hour')['Profit'].sum().reset_index()
    
    # Fill missing hours
    all_hours = pd.DataFrame({'Hour': range(24)})
    hourly_stats = all_hours.merge(hourly_stats, on='Hour', how='left').fillna(0)
    
    # Calculate stats for Hour vs Profit
    x = hourly_stats['Hour']
    y = hourly_stats['Profit']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    correlation, _ = stats.pearsonr(x, y)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create colors based on profit/loss
    colors = ['#00ff88' if p >= 0 else '#ff4444' for p in hourly_stats['Profit']]
    
    ax.bar(hourly_stats['Hour'], hourly_stats['Profit'], color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Regression line
    line_x = np.array([0, 23])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color='#ffaa00', linestyle='--', linewidth=1.5, label='Trend Line', alpha=0.7)
    
    # Add a horizontal line at 0
    ax.axhline(0, color='white', linewidth=0.8, alpha=0.5)
    
    ax.set_title('Profit Distribution by Hour (Trade Close) + Regression', fontsize=16, fontweight='bold', pad=20, color='#e0e0e0')
    ax.set_xlabel('Hour of Day (24h)', fontsize=12, color='#cccccc')
    ax.set_ylabel('Cumulative Profit (USD)', fontsize=12, color='#cccccc')
    ax.set_xticks(range(24))
    ax.grid(axis='y', linestyle='--', alpha=0.2)
    
    # Stats annotation
    stats_text = (f"Correlation: {correlation:.4f}\n"
                  f"P-value: {p_value:.4f}\n"
                  f"Slope: {slope:.2f}")
    
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='#ffaa00'),
            fontsize=11, family='monospace', color='white')

    # Annotate total profit
    total_p = trades_df['Profit'].sum()
    ax.text(0.02, 0.95, f"Total Trades: {len(trades_df)}\nNet Profit: ${total_p:,.2f}", 
            transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5, edgecolor='#00ff88'))

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "profit hour distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

def plot_duration_vs_profit(trades_df):
    print("Generating Trade Duration vs Profit plot...")
    
    # Remove outliers for better visualization if needed, but let's keep all for stats
    x = trades_df['Duration']
    y = trades_df['Profit']
    
    # Calculate stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    correlation, _ = stats.pearsonr(x, y)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Scatter plot with custom aesthetics
    sns.scatterplot(data=trades_df, x='Duration', y='Profit', alpha=0.6, s=60, color='#00d4ff', edgecolor='white', linewidth=0.5, ax=ax)
    
    # Regression line
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color='#ff0077', linewidth=2, label='Regression Line', alpha=0.8)
    
    # Styling
    ax.set_title('Trade Duration vs Profit Analysis', fontsize=16, fontweight='bold', pad=20, color='#e0e0e0')
    ax.set_xlabel('Duration (Hours)', fontsize=12, color='#cccccc')
    ax.set_ylabel('Profit (USD)', fontsize=12, color='#cccccc')
    ax.grid(linestyle=':', alpha=0.3)
    
    # Stats annotation
    stats_text = (f"Correlation: {correlation:.4f}\n"
                  f"R-squared: {r_value**2:.4f}\n"
                  f"P-value: {p_value:.4e}\n"
                  f"Slope: {slope:.4f}")
    
    ax.text(0.98, 0.05, stats_text, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='#ff0077'),
            fontsize=11, family='monospace', color='white')

    # Add 0 line
    ax.axhline(0, color='white', linewidth=0.8, alpha=0.4)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "trade duration vs profit.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

def main():
    try:
        df = load_data(EXCEL_PATH)
        trades_df = process_trades(df)
        
        if trades_df.empty:
            print("No trades found to analyze.")
            return
            
        plot_profit_hour_distribution(trades_df)
        plot_duration_vs_profit(trades_df)
        
        print("\n--- Summary Statistics ---")
        correlation, _ = stats.pearsonr(trades_df['Duration'], trades_df['Profit'])
        _, _, _, p_value, _ = stats.linregress(trades_df['Duration'], trades_df['Profit'])
        print(f"Correlation: {correlation:.4f}")
        print(f"P-value: {p_value:.4e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
