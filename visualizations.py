
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_top_products(df):
    top = df['Description'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title("Top 10 Products")
    return fig

def plot_top_countries(df):
    top = df['Country'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title("Top 10 Countries by Transactions")
    return fig

def plot_daily_orders(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Date'] = df['InvoiceDate'].dt.date
    daily_orders = df.groupby('Date')['InvoiceNo'].nunique()
    fig, ax = plt.subplots(figsize=(12, 4))
    daily_orders.plot(ax=ax)
    ax.set_title("Number of Orders Per Day")
    return fig

def plot_association_rules(rules_df, metric='confidence'):
    top_rules = rules_df.sort_values(by=metric, ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_rules, x=metric, y=top_rules['antecedents'].astype(str) + "->" + top_rules['consequents'].astype(str), ax=ax)
    ax.set_title(f"Top 10 Association Rules by {metric.capitalize()}")
    return fig

def plot_similarity_heatmap(sim_df, top_n=20):
    if sim_df is None or not hasattr(sim_df, 'iloc'):
        return None

    subset = sim_df.iloc[:top_n, :top_n]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(subset, cmap="YlGnBu", ax=ax)
    plt.title("Item Similarity Heatmap")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig
