import pandas as pd

file_path = r"c:\Users\WINDOWS\OneDrive\Desktop\xema data\data\c.kim A set USD.xlsx"
full_df = pd.read_excel(file_path, header=None)

# Find header
header_row_index = -1
for i, row in full_df.iterrows():
    row_values = [str(x).strip() for x in row.values]
    if 'Profit' in row_values and 'Time' in row_values:
        header_row_index = i
        break

df = pd.read_excel(file_path, skiprows=header_row_index + 1, names=full_df.iloc[header_row_index])
df.columns = [str(col).strip() for col in df.columns]
df = df[df['Symbol'].notna()]

print(f"Total deals in file: {len(df)}")
zero_profit = df[df['Profit'] == 0]
nonzero_profit = df[df['Profit'] != 0]

print(f"Deals with 0 profit (Entries/Breakeven): {len(zero_profit)}")
print(f"Deals with non-zero profit (Exits): {len(nonzero_profit)}")

# Check for pairs in exits
nonzero_profit['Time'] = pd.to_datetime(nonzero_profit['Time'])
nonzero_profit = nonzero_profit.sort_values('Time')

# Grouping by 5-second windows
nonzero_profit['Time_5s'] = nonzero_profit['Time'].dt.round('5s')

groups_5s = nonzero_profit.groupby(['Time_5s', 'Direction']).size()
print("\nDistribution of trades per signal group (5-second window):")
print(groups_5s.value_counts())

total_signals_5s = len(groups_5s)
print(f"\nTotal signals with 5s grouping: {total_signals_5s}")
