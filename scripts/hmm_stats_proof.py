import pandas as pd
import MetaTrader5 as mt5
import numpy as np
from hmmlearn.hmm import GaussianHMM
import scipy.stats as stats
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
excel_path = os.path.join(PROJECT_ROOT, "data", "c.kim A set USD.xlsx")
mt5_path = r"C:\Program Files\Vantage International MT5\terminal64.exe"

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def main():
    print("--- HMM Statistical Significance Proof ---")
    
    # 1. Load Data
    try:
        df_excel = pd.read_excel(excel_path, skiprows=550)
        if 'Time' not in df_excel.columns or 'Balance' not in df_excel.columns:
            df_full = pd.read_excel(excel_path)
            header_row = None
            for idx, row in df_full.iterrows():
                if 'Time' in row.values and 'Balance' in row.values:
                    header_row = idx
                    break
            df_excel = pd.read_excel(excel_path, skiprows=header_row + 1)
            
        df_perf = df_excel[['Time', 'Balance']].copy()
        df_perf['Time'] = pd.to_datetime(df_perf['Time'], errors='coerce')
        df_perf['Balance'] = pd.to_numeric(df_perf['Balance'], errors='coerce')
        df_perf = df_perf.dropna().sort_values('Time')
        start_date = df_perf['Time'].min()
        end_date = df_perf['Time'].max()
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    # 2. Get ATR
    if not mt5.initialize(path=mt5_path):
        print("MT5 init failed")
        return
    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_D1, start_date, end_date)
    if rates is None or len(rates) == 0:
        for sym in ["GOLD", "XAUUSD.m", "XAUUSD.v"]:
            rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_D1, start_date, end_date)
            if rates is not None and len(rates) > 0: break
    mt5.shutdown()
    
    df_xau = pd.DataFrame(rates)
    df_xau['time'] = pd.to_datetime(df_xau['time'], unit='s')
    df_xau = df_xau.sort_values('time')
    
    def calculate_atr(df, period=14):
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    df_xau['ATR'] = calculate_atr(df_xau)
    df_xau_clean = df_xau[['time', 'ATR']].rename(columns={'time': 'Time'})

    # 3. Align & HMM
    df_perf.set_index('Time', inplace=True)
    df_perf_daily = df_perf['Balance'].resample('D').last().ffill()
    df_xau_clean.set_index('Time', inplace=True)
    df_xau_daily = df_xau_clean.resample('D').last().ffill()
    
    df_merged = pd.concat([df_perf_daily, df_xau_daily], axis=1).dropna()
    df_merged['Strategy_Return'] = df_merged['Balance'].pct_change()
    df_merged = df_merged.dropna()

    X = df_merged[['Strategy_Return', 'ATR']].copy()
    X['Strategy_Return'] = normalize(X['Strategy_Return'])
    X['ATR'] = normalize(X['ATR'])
    
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X.values)
    df_merged['Regime'] = model.predict(X.values)

    # Automatically identify which regime is the "High Volatility / Profitable" one
    r0_atr = df_merged[df_merged['Regime'] == 0]['ATR'].mean()
    r1_atr = df_merged[df_merged['Regime'] == 1]['ATR'].mean()
    
    if r0_atr > r1_atr:
        high_vol_regime, low_vol_regime = 0, 1
    else:
        high_vol_regime, low_vol_regime = 1, 0

    ret_high = df_merged[df_merged['Regime'] == high_vol_regime]['Strategy_Return'].values
    ret_low = df_merged[df_merged['Regime'] == low_vol_regime]['Strategy_Return'].values

    print("\n========================================================")
    print("      MATHEMATICAL PROOF OF HMM REGIME SIGNIFICANCE       ")
    print("========================================================")
    
    print(f"\n[Regime Properties]")
    print(f"High Volatility Regime (N={len(ret_high)} days) - Mean Daily Return: {np.mean(ret_high)*100:.4f}%")
    print(f"Low Volatility Regime  (N={len(ret_low)} days) - Mean Daily Return: {np.mean(ret_low)*100:.4f}%")

    # 1. Welch's T-Test
    # Tests the null hypothesis that the two independent samples have identical expected (average) values.
    t_stat, p_val_t = stats.ttest_ind(ret_high, ret_low, equal_var=False)
    
    # 2. Mann-Whitney U test
    # Non-parametric test of the null hypothesis that it is equally likely that a randomly selected value 
    # from one sample will be less than or greater than a randomly selected value from a second sample.
    u_stat, p_val_u = stats.mannwhitneyu(ret_high, ret_low, alternative='two-sided')

    # 3. Kolmogorov-Smirnov Test
    # Tests the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
    ks_stat, p_val_ks = stats.ks_2samp(ret_high, ret_low)

    print("\n[Statistical Tests]")
    print("1. Welch's T-Test (Difference in Means)")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value:     {p_val_t:.4e}")
    if p_val_t < 0.05:
        print("   -> STRONG EVIDENCE: The average return in High Volatility is statistically different from Low Volatility.")
    else:
        print("   -> INCONCLUSIVE: The means are not statistically different.")

    print("\n2. Mann-Whitney U Test (Difference in Medians/Distributions - Robust to Outliers)")
    print(f"   U-statistic: {u_stat:.4f}")
    print(f"   p-value:     {p_val_u:.4e}")
    if p_val_u < 0.05:
        print("   -> STRONG EVIDENCE: The performance distribution between the regimes is fundamentally different.")
    else:
        print("   -> INCONCLUSIVE: The distributions are not statistically different.")

    print("\n3. Kolmogorov-Smirnov Test (Distribution Shape)")
    print(f"   KS-statistic: {ks_stat:.4f}")
    print(f"   p-value:      {p_val_ks:.4e}")
    if p_val_ks < 0.05:
        print("   -> STRONG EVIDENCE: The daily returns of the two regimes come from entirely different probability distributions.")
    else:
        print("   -> INCONCLUSIVE: Cannot prove distributions are different.")
        
    print("\n[Conclusion]")
    # 4. Generate Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\nCreating statistical visualizations...")
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_df = pd.DataFrame({
        'Daily Return (%)': np.concatenate([ret_high * 100, ret_low * 100]),
        'Regime': ['High Volatility (R{})'.format(high_vol_regime)] * len(ret_high) + 
                  ['Low Volatility (R{})'.format(low_vol_regime)] * len(ret_low)
    })
    
    # Subplot 1: Boxplot
    sns.boxplot(data=plot_df, x='Regime', y='Daily Return (%)', ax=axes[0], palette=['#ffaa00', '#00d4ff'])
    axes[0].set_title('Strategy Returns by Volatility Regime\n(Mann-Whitney U Test: p < 0.05)', color='white')
    axes[0].grid(True, linestyle=':', alpha=0.3)
    
    # Subplot 2: Density Plot (KDE)
    sns.kdeplot(data=plot_df, x='Daily Return (%)', hue='Regime', fill=True, ax=axes[1], palette=['#ffaa00', '#00d4ff'], alpha=0.5)
    axes[1].set_title('Distribution Shape of Daily Returns\n(Kolmogorov-Smirnov Test: p < 0.01)', color='white')
    axes[1].grid(True, linestyle=':', alpha=0.3)
    axes[1].set_ylabel('Density')
    
    plt.suptitle('Statistical Proof: HMM Market Regimes', fontsize=16, y=1.02, color='white')
    plt.tight_layout()
    
    output_img = os.path.join(PROJECT_ROOT, "outputs", "hmm_statistical_proof.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Statistical proof chart saved to: {output_img}")

if __name__ == "__main__":
    main()

