from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Retail Recommendation Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def save_plot(fig, filename):
    path = os.path.join("reports", filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

def generate_pdf_report(df, rules=None, sim_df=None, cleaning_options=None, filename="retail_report.pdf", general_plots=None, recommendation_plots = None, evaluation_stats=None):
    os.makedirs("reports", exist_ok=True)
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    # 1. Data Preview
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Data Preview (First 5 Rows):", ln=True)
    pdf.set_font("Courier", size=9)
    preview = df.head().to_string(index=False)
    for line in preview.split('\n'):
        pdf.multi_cell(0, 5, line)
    pdf.ln(5)

    # 2. Data Exploration
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Data Exploration:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, f"Shape: {df.shape}")
    pdf.multi_cell(0, 8, f"Missing values:\n{df.isnull().sum().to_string()}")
    pdf.ln(5)

    # 3. Visualizations
    if general_plots:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "3. Visualizations:", ln=True)
        for title, img_path in general_plots.items():
            pdf.cell(0, 10, f"{title}", ln=True)
            pdf.image(img_path, w=170)
            pdf.ln(5)
        
    # 4. Cleaning Applied
    if cleaning_options:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "4. Cleaning Steps Applied:", ln=True)
        pdf.set_font("Arial", size=11)
        for opt in cleaning_options:
            pdf.cell(0, 10, f"- {opt}", ln=True)
        pdf.ln(5)

    # 5. Recommendations
    if rules is not None and not rules.empty:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "5. Applied Recommendations (Top Rules):", ln=True)
        pdf.set_font("Arial", size=11)
        for idx, row in rules.head(5).iterrows():
            pdf.multi_cell(0, 10,f"  - {set(row['antecedents'])} -> {set(row['consequents'])} | Conf: {row['confidence']:.2f}")
        pdf.ln(5)

    if sim_df is not None:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Top 5 Item Similarities:", ln=True)
        pdf.set_font("Arial", size=11)
        for item in sim_df.columns[:5]:
            similar = sim_df[item].nlargest(5).drop(item)
            sims = ", ".join([f"{i}: {v:.2f}" for i, v in similar.items()])
            pdf.multi_cell(0, 10, f"  - {item} -> {sims}")
        pdf.ln(5)

    # 6. Visual Results of Recommendations
    if recommendation_plots:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "6. Recommendation Results (Visual):", ln=True)
        for title, img_path in recommendation_plots.items():
            pdf.multi_cell(0, 10, f" {title}")
            pdf.image(img_path, w=170)
            pdf.ln(5)

    # 7. Evaluation of Recommendations
    if evaluation_stats:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "7. Evaluation of Recommendations", ln=True)
        pdf.set_font("Arial", size=11)

        if "total_rules" in evaluation_stats:
            pdf.multi_cell(0, 10, f"- Total Rules: {evaluation_stats['total_rules']}")
        if "high_confidence_rules" in evaluation_stats:
            pdf.multi_cell(0, 10, f"- High Confidence Rules (confidence > 0.8): {evaluation_stats['high_confidence_rules']}")
        if "avg_lift" in evaluation_stats:
            pdf.multi_cell(0, 10, f"- Average Lift: {evaluation_stats['avg_lift']:.2f}")
        if "similarity_hist_path" in evaluation_stats:
            pdf.multi_cell(0, 10, "- Similarity Score Distribution (excluding self-similarity):")
            pdf.image(evaluation_stats["similarity_hist_path"], w=170)


    # 8. Security & Data Notice
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "8. Data Security & Disclaimer", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, 
        "- No personal or sensitive data is stored.\n"
        "- All processing happens locally during the session.\n"
        "- Uploaded files are only used temporarily in-memory.\n"
        "- Generated reports are saved locally on your system.\n"
        "- This tool is intended for educational or non-sensitive use cases only."
    )

    output_path = os.path.join("reports", filename)
    pdf.output(output_path)
    return output_path
