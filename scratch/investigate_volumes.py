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

# Non-zero profit deals (exits)
exits = df[df['Profit'] != 0].copy()
print(f"Total Exit Deals: {len(exits)}")
print("\nVolume Distribution in Exits:")
print(exits['Volume'].value_counts())

# Check for the 0.4 and 0.2 pairs
exits['Date'] = pd.to_datetime(exits['Time']).dt.date
exits['Hour'] = pd.to_datetime(exits['Time']).dt.hour

# Let's see how many 0.4 and 0.2 exist
v04 = len(exits[exits['Volume'] == 0.4])
v02 = len(exits[exits['Volume'] == 0.2])
print(f"\n0.4 trades: {v04}")
print(f"0.2 trades: {v02}")

# Let's try to group by something that identifies the signal
# Maybe they share a Comment or happen very close in entry time?
# Wait, let's look at the entries for those exits.
# Actually, let's just group by Time and Direction and see the volumes.
exits['Time_str'] = exits['Time'].astype(str)
groups = exits.groupby(['Time_str', 'Direction']).agg({
    'Volume': list,
    'Profit': 'sum'
})

print("\nSample Groups and their Volumes:")
print(groups.head(10))

print("\nUnique Volume combinations in groups:")
print(groups['Volume'].apply(lambda x: sorted(x)).astype(str).value_counts())
