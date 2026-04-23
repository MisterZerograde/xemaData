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

# Let's try to link Deals by their Order ID if possible?
# Standard MT5 reports have 'Order' column which is the ticket of the order.
# Each deal belongs to an order.
# But 0.4 and 0.2 are separate orders.
# Do they share a comment? Or a 'Magic' (not in Excel usually)?

# Let's look at the sequence of all deals.
df['Time_dt'] = pd.to_datetime(df['Time'])
df = df.sort_values(['Time_dt', 'Deal'])

# If we just take the sequence of entries and exits, maybe we can pair them.
# But the user says 0.4 and 0.2 is one signal.
# Let's see if they occur together in the deal list.
# I'll try to find the 'in' deals (entries).
entries = df[df['Profit'] == 0].copy()
exits = df[df['Profit'] != 0].copy()

print(f"Total Entries: {len(entries)}")
print(f"Total Exits: {len(exits)}")

# Group entries by time
entry_groups = entries.groupby(['Time', 'Direction']).size()
print("\nEntries per unique time/direction:")
print(entry_groups.value_counts())
