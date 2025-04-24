# Retail Store Data Explorer & Recommender System



## Features
This Streamlit app lets you upload and explore retail data, clean it, see useful charts, get product recommendations, and download a report. The key features of this application are was follows:

- Upload data from file or URL
- Data exploration (info, missing values, summary stats)
- Customizable data cleaning (drop nulls, filter invalid rows, etc.)
- Visualizations: top products, countries, daily orders
- Recommendation Engine:
  - **Association Rule Mining** (Apriori algorithm via `mlxtend`)
  - **Collaborative Filtering** (Cosine similarity between products)
- Heatmaps and metric plots
- PDF report export with visual and analytical summaries
- Local-only data handling for privacy


## System Requirement 
```bash
- OS: Windows, macOS, or Linux
- Memory: Minimum 45 GB RAM (64 GB recommended for large datasets)
- CPU: At least 4 CPU cores
- Internet access (only if loading datasets from URLs)
- python: Python 3+
- github
```

## Installation and Required Dependencies

Clone the repository and install dependencies:
```bash
git clone https://github.com/unishajoshi/recommendation.git
cd deep-fake-detection
pip install -r requirements.txt
```

## Execute the command

Run the Streamlit app:
```bash
python -m steamlit run recommender_user_interface.py
```

Then open your browser to:
```
http://localhost:8501/
```

## Sample Dataset

You can test the app using the UCI Online Retail Dataset:
- "Online Retail.CSV"

Make sure to convert Excel to CSV if needed.


## Acknowledgements

Developed by **Unisha Joshi** as part of a data product assignment.

---

**Note**: This app does not store or transmit any uploaded data. It is intended for educational and academic use.
