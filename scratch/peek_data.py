import pandas as pd

file_path = r"c:\Users\WINDOWS\OneDrive\Desktop\xema data\data\c.kim A set USD.xlsx"
full_df = pd.read_excel(file_path, header=None)

# Dynamic header search
header_row_index = -1
for i, row in full_df.iterrows():
    row_values = [str(x).strip() for x in row.values]
    if 'Profit' in row_values and 'Time' in row_values:
        header_row_index = i
        break

if header_row_index != -1:
    df = pd.read_excel(file_path, skiprows=header_row_index + 1, names=full_df.iloc[header_row_index])
    df.columns = [str(col).strip() for col in df.columns]
    print("Comments unique values:\n", df['Comment'].unique()[:20])
    print("\nSample rows:\n", df[['Time', 'Type', 'Profit', 'Comment']].head(10))
else:
    print("Header not found")
