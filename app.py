import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
import os

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Personal Key OpenAI

st.set_page_config(page_title="Excel Insight Engine", layout="wide")

st.title("📊 Excel Insight Engine")
st.subheader("Upload any CSV or Excel file and get automatic insights")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Load CSV or Excel
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    # Clean up: drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Basic Preview
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Basic Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write(df.dtypes)

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Descriptive Stats")
    st.write(df.describe())

    # Correlation Matrix
    st.write("### 🔗 Correlation Matrix")
    corr = df.corr(numeric_only=True)

    if corr.empty:
        st.warning("No numeric columns found to compute correlation.")
    else:
        st.dataframe(corr)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # Column Insights
    st.write("### 🔍 Column Insights")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()

    selected_col = st.selectbox("Select any column to analyze", df.columns)

    st.write(f"**Column Type:** {df[selected_col].dtype}")
    st.write(f"**% Missing:** {100 * df[selected_col].isnull().mean():.2f}%")
    st.write(f"**# Unique Values:** {df[selected_col].nunique()}")

    if selected_col in num_cols:
        st.write("#### 📊 Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_col], ax=ax)
        st.pyplot(fig)

        st.write("### 🚨 Outlier Detection (Z-score > 3)")
        z_score = (df[selected_col] - df[selected_col].mean()) / df[selected_col].std()
        outliers = df[z_score.abs() > 3]
        st.write(f"Outliers found: {len(outliers)}")
        st.dataframe(outliers)

    elif selected_col in cat_cols:
        st.write("#### 🔢 Top 10 Value Counts")
        st.write(df[selected_col].value_counts().head(10))

    # Data Quality Report
    st.write("### 🧼 Data Quality Report")
    missing_total = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]

    st.write(f"**Total Missing Values:** {missing_total}")
    st.write(f"**Duplicate Rows:** {duplicate_rows}")
    st.write(f"**Constant Columns:** {', '.join(constant_cols) if constant_cols else 'None'}")

    mixed_types = []
    for col in df.columns:
        types = df[col].map(type).nunique()
        if types > 1:
            mixed_types.append(col)
    st.write(f"**Mixed-Type Columns:** {', '.join(mixed_types) if mixed_types else 'None'}")

    # AI SUMMARY
    st.write("### 🧠 AI Summary")

    if st.button("Generate Summary with AI"):
        prompt = f"""You are a data analyst. Provide a clear, concise summary of the dataset.

           - Data shape: {df.shape[0]} rows × {df.shape[1]} columns
           - Missing values:\n{df.isnull().sum().to_string()}
           - Descriptive stats:\n{df.describe().to_string()}
           - Correlation matrix:\n{df.corr(numeric_only=True).to_string()}
           """

        with st.spinner("Generating summary..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.7
                )

                summary = response.choices[0].message.content
                st.success("Summary generated!")
                st.write(summary)

            except Exception as e:
                if "RateLimitError" in str(e) or "quota" in str(e).lower():
                    st.error(
                        "❌ You've hit your OpenAI quota. Visit https://platform.openai.com/account/usage to check your limits.")
                else:
                    st.error(f"❌ Something went wrong: {e}")