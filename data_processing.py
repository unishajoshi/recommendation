
import pandas as pd

def clean_data(df, options):
    if "Drop missing CustomerID or Description" in options:
        df = df.dropna(subset=["CustomerID", "Description"])
    if "Remove canceled transactions (InvoiceNo starts with 'C')" in options and "InvoiceNo" in df.columns:
        df = df[~df['InvoiceNo'].astype(str).str.startswith("C")]
    if "Filter Quantity and UnitPrice > 0" in options:
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    if "Clean Description text" in options:
        df["Description"] = df["Description"].str.lower().str.strip()
    if "Convert InvoiceDate to datetime" in options:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    if "Convert CustomerID to string" in options:
        df["CustomerID"] = df["CustomerID"].astype(str)
    return df
