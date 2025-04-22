from streamlit.runtime.scriptrunner import add_script_run_ctx
import os
import numpy as np
from difflib import get_close_matches
import streamlit as st
import shutil
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import clean_data
from visualizations import plot_top_products, plot_top_countries, plot_daily_orders,plot_similarity_heatmap, plot_association_rules
from recommender import generate_association_rules,recommend_from_rules,generate_similarity_matrix,build_similarity_scores,recommend_similar_items
from report_generator import generate_pdf_report, save_plot
from logger import log_action

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Retail Data Explorer & Recommender", layout="wide")
st.title("🛍️ Retail Store Data Explorer & Recommender System")
st.markdown("👨‍💻 By: Unisha Joshi")


# Load data
with st.sidebar.expander("❓ Help & Instructions", expanded=False):
    st.markdown("""
    ### 📥 Loading Data
    - Upload a CSV file from your computer or provide a public URL.
    - Dataset must include columns like `InvoiceNo`, `Description`, `Quantity`, `UnitPrice`, and `CustomerID`.

    ### 🔍 Exploring Data
    - View the first few rows of data.
    - Expand **Data Exploration** to check data types, null values, and summary stats.

    ### 🧼 Cleaning Data
    - Select options to filter and clean rows (e.g., drop nulls, remove canceled transactions).
    - Click **Apply Cleaning** to update the dataset.

    ### 🤖 Recommendations
    - Generate **association rules** to suggest frequently bought-together items.
    - Use **collaborative filtering** to find similar items based on purchase patterns.

    ### 📈 Visualizations
    - Automatically shows top products, countries, and daily order trends.
    - Recommender outputs (rules + similarities) are visualized in charts and heatmaps.

    ### 📝 Exporting Report
    - Enter a report filename and click **Generate PDF Report**.
    - Download your personalized insights, visualizations, and recommendations.

    ### 🔄 Resetting the App
    - Use the **Reset Session** button before exiting to clear memory and remove temporary files.

    ### 🔐 Security
    - Data is never stored or sent to external servers.
    - All processing happens locally in your session.
    """)
with st.sidebar.expander("🔐 Security Notice"):
    st.markdown("""
    - Uploaded files are **not stored** permanently.
    - All data is processed locally during your session.
    - Please avoid uploading personally identifiable or sensitive data.
    - The app is intended for **educational and analytical purposes** only.
    - All generated reports are saved locally in your browser or system.
    """)

st.sidebar.header("📥 Load Dataset")
upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
url_input = st.sidebar.text_input("Or load from URL")

query_params = st.query_params
if "reset" in query_params:
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Delete all generated files
    if os.path.exists("reports"):
        shutil.rmtree("reports")
        os.makedirs("reports")

    # Redirect back to clean app state (remove the ?reset param)
    st.query_params.clear()  # clears all params
    st.success("Session fully reset.")
    st.info("👈 Upload a file or provide a URL to begin.")
    st.markdown("""
        **Example Dataset URL: Online Retail CSV (UCI Repository)**  
        
        Link: https://archive.ics.uci.edu/static/public/352/data.csv

        You can **paste this link** into the URL input box on the sidebar to test the app.

        OR

        You can download the excel file from the link below and upload it from the **Browse files** button on the side bar.
        
        Link: https://archive.ics.uci.edu/dataset/352/online+retail
        
        <span style='color:orange'> ⚠️Note: Please change the file format to .csv before uploading.</span>
    """,unsafe_allow_html=True)
    st.stop()  # stop rerun from processing further

df = None
if upload:
    df = pd.read_csv(upload)
    st.success("Loaded from file.")
    log_action("Data Load", "SUCCESS", "Loaded from uploaded file.")
elif url_input:
    try:
        df = pd.read_csv(url_input)
        st.success("Loaded from URL.")
        log_action("Data Load", "SUCCESS", "Loaded from URL.")
    except:
        st.error("Failed to load data from URL.")
        log_action("Data Load", "SUCCESS", "Loaded from URL.")

if df is not None:
    # Data Preview section
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())
    log_action("Data Preview", "SHOWN", "Displayed first 5 rows.")

    # Data Exploration Section
    st.subheader("📊 Data Exploration")
    with st.expander("Show Info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        log_action("Data Exploration", "SHOWN", "Displayed info summary.")
    with st.expander("Missing Values"):
        st.write(df.isnull().sum())
        log_action("Data Exploration", "SHOWN", "Displayed missing values.")
    with st.expander("Summary Stats"):
        st.write(df.describe(include='all'))
        log_action("Data Exploration", "SHOWN", "Displayed descriptive statistics.")    

    # Data Visualization section
    st.subheader("📈 Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        if "Description" in df.columns:
            st.pyplot(plot_top_products(df))
            log_action("Visualization", "SUCCESS", "Top products plot rendered.")
    with col2:
        if "Country" in df.columns:
            st.pyplot(plot_top_countries(df))
            log_action("Visualization", "SUCCESS", "Top countries plot rendered.")

    if "InvoiceDate" in df.columns:
        st.pyplot(plot_daily_orders(df))
        log_action("Visualization", "SUCCESS", "Daily orders plot rendered.")

    # Data Cleaning Section
    st.subheader("🧼 Data Cleaning Options")    
    options = st.multiselect("Select cleaning options", [
        "Drop missing CustomerID or Description",
        "Remove canceled transactions (InvoiceNo starts with 'C')",
        "Filter Quantity and UnitPrice > 0",
        "Clean Description text",
        "Convert InvoiceDate to datetime",
        "Convert CustomerID to string"
    ])
    if st.button("Apply Cleaning"):
        df = clean_data(df, options)
        st.success("Cleaning applied.")
        st.write(df.head())
        log_action("Data Cleaning", "SUCCESS", f"Options: {options}")
        

    # Recommendation Section
    st.subheader("🤖 Generate Recommendations")

    if st.button("Generate Association Rules"):
        rules = generate_association_rules(df)
        st.session_state.rules = rules
        st.success("Association rules generated.")
        log_action("Association Rules", "SUCCESS", f"{len(rules)} rules generated.")
    # Association Rule Plot
    if "rules" in st.session_state:
        st.subheader("📊 Association Rule Metrics")
        metric = st.selectbox("Select metric to plot", ["confidence", "support", "lift"])
        st.pyplot(plot_association_rules(st.session_state.rules, metric))
    if "rules" in st.session_state and "Description" in df.columns:
        st.subheader("🧩 Item-wise Association Rule Recommendations")
    
        product_list = sorted(df['Description'].dropna().unique())
        selected_item = st.selectbox("Select a product to get association-based recommendations", product_list, index=None, placeholder="Choose product...")
    
        if selected_item:
            recs = recommend_from_rules(st.session_state.rules, selected_item)
            if not recs.empty:
                st.write(f"💡 Recommended with **{selected_item}**:")
                st.dataframe(recs)
            else:
                st.warning("No rules found for this product.")
    
    # --- Evaluate Association Rules ---
    if "rules" in st.session_state and not st.session_state.rules.empty:
        rules_df = st.session_state.rules
        high_conf = rules_df[rules_df['confidence'] > 0.8]
        avg_lift = rules_df['lift'].mean()
    
        st.markdown("### 🔍 Association Rule Evaluation")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rules", len(rules_df))
        col2.metric("High Confidence Rules (> 0.8)", len(high_conf))
        col3.metric("Avg. Lift", f"{avg_lift:.2f}")
    
        st.write("**Confidence Distribution**")
        st.bar_chart(rules_df['confidence'])

        
    # --- SIMILARITY MATRIX SECTION ---
    st.subheader("📊 Item Similarity (Collaborative Filtering)")
    
    # Button to generate similarity matrix
    if st.button("Generate Similarity Matrix"):
        sim_df = generate_similarity_matrix(df)
        st.session_state.sim_df = sim_df
        st.success("Similarity matrix generated.")
        log_action("Similarity Matrix", "SUCCESS", f"Shape: {sim_df.shape}")
    
    # Show heatmap if matrix is generated
    if "sim_df" in st.session_state and st.session_state.sim_df is not None:
        st.subheader("🔥 Item Similarity Heatmap")
        fig5 = plot_similarity_heatmap(st.session_state.sim_df)
    
        if fig5:
            st.pyplot(fig5)
    
            # Save plot for report (if you're using recommendation_plots)
            if "recommendation_plots" not in st.session_state:
                st.session_state.recommendation_plots = {}
    
            st.session_state.recommendation_plots["Similarity Heatmap"] = save_plot(fig5, "similarity_heatmap.png")
        else:
            st.warning("Could not generate heatmap from the similarity matrix.")

    # ---------- ITEM-WISE SIMILARITY RECOMMENDATIONS ----------
    if "sim_df" in st.session_state and st.session_state.sim_df is not None:
        st.subheader("🧭 Item-wise Similarity Recommendations")
    
        similarity_item = st.selectbox("Select a product for similarity-based recommendations", list(st.session_state.sim_df.columns), index=None, placeholder="Choose product...")
    
        if similarity_item:
            similar_items = (
                st.session_state.sim_df[similarity_item]
                .drop(similarity_item, errors="ignore")
                .sort_values(ascending=False)
                .head(5)
            )
            st.write(f"🔍 Top similar items to **{similarity_item}**:")
            st.dataframe(pd.DataFrame({
                "Similar Item": similar_items.index,
                "Similarity Score": similar_items.values
            }))
    # --- Evaluate Similarity Matrix ---
    if "sim_df" in st.session_state and st.session_state.sim_df is not None:
        st.markdown("### 🔍 Similarity Matrix Evaluation")
    
        # Flatten matrix to get score distribution
        sim_vals = st.session_state.sim_df.values.flatten()
        sim_vals = sim_vals[sim_vals != 1]  # exclude diagonal/self similarity  
        st.write("**Histogram of Similarity Scores**")
        fig, ax = plt.subplots()
        ax.hist(sim_vals, bins=30, color='skyblue', edgecolor='black')
        ax.set_title("Distribution of Similarity Scores")
        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)


    # Prepare plots
    general_plots = {}
    recommendation_plots = {}
    
    if "Description" in df.columns:
        fig1 = plot_top_products(df)
        general_plots["Top Products"] = save_plot(fig1, "top_products.png")
    
    if "Country" in df.columns:
        fig2 = plot_top_countries(df)
        general_plots["Top Countries"] = save_plot(fig2, "top_countries.png")
    
    if "InvoiceDate" in df.columns:
        fig3 = plot_daily_orders(df)
        general_plots["Daily Orders"] = save_plot(fig3, "daily_orders.png")
    
    if "rules" in st.session_state:
        fig4 = plot_association_rules(st.session_state.rules, "confidence")
        recommendation_plots["Association Rule Plot"] = save_plot(fig4, "association_plot.png")
    
    if "sim_df" in st.session_state:
        fig5 = plot_similarity_heatmap(st.session_state.sim_df)
        if fig5:
            recommendation_plots["Similarity Heatmap"] = save_plot(fig5, "similarity_heatmap.png")
               
    # ------------------- REPORT SECTION -------------------
    st.subheader("📝 Export Report")
    report_name = st.text_input("Report filename", value="retail_report.pdf")
    
    if st.button("Generate PDF Report"):
        if df is not None:
            cleaning_applied = options if 'options' in locals() else []
            rules = st.session_state.get("rules")
            sim_df = st.session_state.get("sim_df")
    
            # --- Evaluation: Association Rules ---
            evaluation_stats = {}
            if rules is not None and not rules.empty:
                high_conf = rules[rules["confidence"] > 0.8]
                avg_lift = rules["lift"].mean()
                evaluation_stats["total_rules"] = len(rules)
                evaluation_stats["high_confidence_rules"] = len(high_conf)
                evaluation_stats["avg_lift"] = avg_lift
    
            # --- Evaluation: Similarity Matrix ---
            hist_path = None
            if sim_df is not None:
                import numpy as np
                sim_vals = sim_df.values.flatten()
                sim_vals = sim_vals[sim_vals < 0.999]
                top_scores = np.sort(sim_vals)[-50:]
    
                # Save histogram
                fig, ax = plt.subplots()
                ax.hist(sim_vals, bins=30, color='skyblue', edgecolor='black')
                ax.set_title("Distribution of Similarity Scores")
                ax.set_xlabel("Similarity Score")
                ax.set_ylabel("Frequency")
                plt.tight_layout()
                hist_path = "reports/similarity_score_hist.png"
                fig.savefig(hist_path)
                plt.close()
    
                evaluation_stats["similarity_hist_path"] = hist_path
                evaluation_stats["similarity_top_scores"] = top_scores.tolist()
    
            # --- Generate Report ---
            report_path = generate_pdf_report(
                df,
                rules=rules,
                sim_df=sim_df,
                cleaning_options=cleaning_applied,
                filename=report_name,
                general_plots=general_plots,
                recommendation_plots=recommendation_plots,
                evaluation_stats=evaluation_stats  # ⬅️ pass evaluation into the PDF generator
            )

        with open(report_path, "rb") as f:
            st.download_button("📥 Download Report", f, file_name=report_name)
        log_action("Report Export", "SUCCESS", f"Report saved as {report_name}")
    else:
        st.warning("Please load and clean a dataset first.")
        log_action("Report Export", "Fail", f"Failed to save report as {report_name}")

    # Addressing Security
    st.markdown("---")
     
    if st.button("🔄 Reset Session"):
        # Clear Streamlit session state
        
       for key in list(st.session_state.keys()):
           del st.session_state[key]
           if os.path.exists("reports"):
               shutil.rmtree("reports")
               os.makedirs("reports", exist_ok=True)
               st.success("Session has been reset and all generated files were deleted.")
               st.query_params.update({"reset": "true"})
               st.rerun()

    st.info("🔒 **Reminder:** Use the Reset Session button to clear memory and files before exiting.")


else:
    st.info("👈 Upload a file or provide a URL to begin.")
    st.markdown("""
        **Example Dataset URL: Online Retail CSV (UCI Repository)**  
        
        Link: https://archive.ics.uci.edu/static/public/352/data.csv

        You can **paste this link** into the URL input box on the sidebar to test the app.

        OR

        You can download the excel file from the link below and upload it from the **Browse files** button on the side bar.
        
        Link: https://archive.ics.uci.edu/dataset/352/online+retail
        
        <span style='color:orange'> ⚠️Note: Please change the file format to .csv before uploading.</span>
    """,unsafe_allow_html=True)
