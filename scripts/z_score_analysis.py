import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_z_score(outcomes):
    if len(outcomes) < 2: return 0, 0, 0, 0, 0
    N = len(outcomes)
    W = sum(outcomes)
    L = N - W
    if W == 0 or L == 0: return 0, W, L, 1, 0
    runs = 1
    for i in range(1, len(outcomes)):
        if outcomes[i] != outcomes[i-1]:
            runs += 1
    E_R = (2.0 * W * L / N) + 1
    V_R = (2.0 * W * L * (2.0 * W * L - N)) / (N**2 * (N - 1))
    Z = (runs - E_R) / np.sqrt(V_R) if V_R > 0 else 0
    return Z, W, L, runs, E_R

# Load
file_path = r"c:\Users\WINDOWS\OneDrive\Desktop\xema data\data\c.kim A set USD.xlsx"
full_df = pd.read_excel(file_path, header=None)
header_row_index = -1
for i, row in full_df.iterrows():
    row_values = [str(x).strip() for x in row.values]
    if 'Profit' in row_values and 'Time' in row_values:
        header_row_index = i
        break
df = pd.read_excel(file_path, skiprows=header_row_index + 1, names=full_df.iloc[header_row_index])
df.columns = [str(col).strip() for col in df.columns]
df = df[df['Symbol'].notna()]

# All Exits
exits = df[df['Profit'] != 0].copy()
exits['Time'] = pd.to_datetime(exits['Time'])
exits = exits.sort_values('Time')

# Logic to group into 120 signals
# We have 120 of 0.4 and 120 of 0.2. 
# We'll create "slots" for 120 signals and fill them.
signals_data = []
temp_04 = exits[exits['Volume'] == 0.4].to_dict('records')
temp_02 = exits[exits['Volume'] == 0.2].to_dict('records')

# Since it's a Duo, the 0.4 and 0.2 are entered together.
# We'll pair them based on their appearance in the list (chronological order)
# since they are likely executed in the same sequence.
for i in range(120):
    p1 = temp_04[i]
    p2 = temp_02[i]
    
    total_profit = p1['Profit'] + p2['Profit']
    signals_data.append({
        'Time': max(p1['Time'], p2['Time']), # Signal closes when the last part closes
        'Profit': total_profit,
        'Outcome': 1 if total_profit > 0 else 0
    })

signals = pd.DataFrame(signals_data).sort_values('Time').reset_index(drop=True)

# Stats
z_real, w, l, r, er = calculate_z_score(signals['Outcome'].tolist())
win_rate = (w / 120) * 100

# Plot
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2.5, 1]})

signals['Equity'] = signals['Profit'].cumsum()
ax1.plot(signals['Equity'].values, color='#00d4ff', linewidth=2.5, label='Signal Equity')
ax1.fill_between(range(len(signals)), signals['Equity'], color='#00d4ff', alpha=0.1)

# Markers
wins = signals[signals['Outcome'] == 1]
losses = signals[signals['Outcome'] == 0]
ax1.scatter(wins.index, wins['Equity'], color='#00ff88', s=40, label='Win Signal (Duo)', zorder=5)
ax1.scatter(losses.index, losses['Equity'], color='#ff4444', s=40, label='Loss Signal (Duo)', zorder=5)

ax1.set_title(f'Final Z-Score Analysis: 120 Duo Signals (0.4 + 0.2)', fontsize=18, pad=20, fontweight='bold')
ax1.set_ylabel('Total Profit (USD)', fontsize=13)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend()

stats_text = (
    f"📊 SIGNAL PERFORMANCE (DUO GROUPING)\n"
    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"Total Signals:  {len(signals):<15} |  Actual Runs:    {r}\n"
    f"Win Signals:    {w:<15} |  Expected Runs:  {er:.2f}\n"
    f"Loss Signals:   {l:<15} |  Win Rate:       {win_rate:.1f}%\n"
    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"REAL Z-SCORE:  {z_real:+.4f}\n"
    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"Verdict: {'STREAKING (Trending)' if z_real < -2 else 'ALTERNATING (Mean-Reverting)' if z_real > 2 else 'RANDOM (No dependency)'}"
)

ax2.text(0.5, 0.5, stats_text, ha='center', va='center', family='monospace', fontsize=13,
         bbox=dict(facecolor='#1a1a1a', edgecolor='#00d4ff', boxstyle='round,pad=1.5', alpha=1.0))
ax2.axis('off')

plt.tight_layout()
output_path = r"c:\Users\WINDOWS\OneDrive\Desktop\xema data\artifacts\final_duo_z_score.png"
plt.savefig(output_path, dpi=300)

print(f"--- FINAL DUO ANALYSIS ---")
print(f"Signals: {len(signals)}")
print(f"Wins: {w} | Losses: {l}")
print(f"Z-Score: {z_real:.4f}")
