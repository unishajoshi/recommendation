
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics.pairwise import cosine_similarity

def generate_association_rules(df, top_n_items=500):
    # Limit to top N items to reduce memory usage
    top_items = df['Description'].value_counts().nlargest(top_n_items).index
    df_filtered = df[df['Description'].isin(top_items)]

    # Create basket format
    basket = df_filtered.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Run Apriori
    frequent_itemsets = apriori(basket_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    return rules.sort_values(by="confidence", ascending=False)

def recommend_from_rules(rules, item_name, top_n=5):
    filtered = rules[rules['antecedents'].apply(lambda x: item_name in x)]
    recommendations = filtered.sort_values(by='confidence', ascending=False).head(top_n)
    return recommendations[['antecedents', 'consequents', 'confidence', 'lift','support']]

def generate_similarity_matrix(df):
    user_item_matrix = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0)
    item_user_matrix = user_item_matrix.T
    similarity = cosine_similarity(item_user_matrix)
    sim_df = pd.DataFrame(similarity, index=item_user_matrix.index, columns=item_user_matrix.index)
    return sim_df

def build_similarity_scores(sim_df, top_n=5):
    all_scores = []

    for item in sim_df.columns:
        similar_items = sim_df[item].drop(item, errors="ignore").sort_values(ascending=False).head(top_n)
        for sim_item, score in similar_items.items():
            all_scores.append({
                "Product": item,
                "Similar Item": sim_item,
                "Similarity": round(score, 4)
            })

    return pd.DataFrame(all_scores)

def recommend_similar_items(sim_df, item_name, top_n=5):
    if item_name not in sim_df.columns or item_name not in sim_df.index:
        return pd.DataFrame({'message': [f"Item '{item_name}' not found in similarity matrix."]})
    similar_items = sim_df[item_name].drop(item_name, errors='ignore').sort_values(ascending=False).head(top_n)
    return pd.DataFrame({'Item': similar_items.index, 'Similarity': similar_items.values})

