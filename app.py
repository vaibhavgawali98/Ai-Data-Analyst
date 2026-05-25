import streamlit as st
import pandas as pd

# importing functions from other files
from utils.cleaning import (
    get_missing_values,
    remove_duplicates,
    fill_missing_values
)

from utils.visualization import (
    plot_histogram,
    plot_boxplot,
    plot_correlation_heatmap,
    plot_scatter
)

from utils.insights import (
    generate_basic_insights
)

from utils.ml_models import (
    train_classification_model
)

# HOME PAGE
st.set_page_config(
    page_title="SmartEDA",
    page_icon="📊",
    layout="wide"
)

#title
st.title("📊 SmartEDA")
st.write(
    "Upload your CSV dataset and get automatic analysis."
)


# File Upload
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ---------------------------------------------------
# Main App

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    
    # Dataset preview
    st.subheader("📁 Dataset Preview")

    st.dataframe(df.head())


    # Dataset Shape
    st.subheader("📌 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])


    # Remove Duplicates Button
    if st.button("Remove Duplicates"):

        original_shape = df.shape[0]

        df = remove_duplicates(df)

        new_shape = df.shape[0]

        removed = original_shape - new_shape

        st.success(f"{removed} duplicate rows removed.")
    
    # Missing values
    st.subheader("⚠ Missing Values")

    missing_df = get_missing_values(df)

    st.dataframe(missing_df)

    # Handling missing values
    st.subheader("🧹 Handle Missing Values")

    columns_with_missing = df.columns[
        df.isnull().sum() > 0
    ].tolist()

    if columns_with_missing:

        selected_missing_column = st.selectbox(
            "Select Column",
            columns_with_missing
        )

        method = st.selectbox(
            "Select Filling Method",
            ["Mean", "Median", "Mode"]
        )

        if st.button("Fill Missing Values"):

            df = fill_missing_values(
                df,
                selected_missing_column,
                method
            )

            st.success(
                f"Missing values filled using {method}."
            )

    else:

        st.success("No missing values found.")

    
    # Numerical Columns
    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    
    # Statistical Summary
    if numerical_cols:

        st.subheader("📈 Statistical Summary")

        st.dataframe(df[numerical_cols].describe())

    # -------------------------------------------
    # Visualization
    # -------------------------------------------

        st.subheader("📊 Data Visualization")

        selected_column = st.selectbox(
            "Select Numerical Column",
            numerical_cols
        )

        # Create Charts
        histogram_fig = plot_histogram(df, selected_column)

        boxplot_fig = plot_boxplot(df, selected_column)

        # Two Charts In One Row
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(histogram_fig)

        with col2:
            st.pyplot(boxplot_fig)

        # -------------------------------------------
        # Scatter Plot + Heatmap
        # -------------------------------------------

        if len(numerical_cols) >= 2:

            st.subheader("📈 Relationship Analysis")

            col_x, col_y = st.columns(2)

            with col_x:

                x_column = st.selectbox(
                    "Select X-axis Column",
                    numerical_cols
                )

            with col_y:

                y_column = st.selectbox(
                    "Select Y-axis Column",
                    numerical_cols
                )

            # Create Charts
            scatter_fig = plot_scatter(
                df,
                x_column,
                y_column
            )

            heatmap_fig = plot_correlation_heatmap(df)

            # Two Charts In One Row
            col3, col4 = st.columns(2)

            with col3:
                st.pyplot(scatter_fig)

            with col4:
                st.pyplot(heatmap_fig)

        else:

            st.warning(
                "At least 2 numerical columns are required for scatter plot."
            )
    else:
        st.warning("No numerical columns found.")

    # -----------------------------------------------
    # AI Insights
    # -----------------------------------------------

    st.subheader("🧠 AI Generated Insights")

    insights = generate_basic_insights(df)

    for insight in insights:

        st.info(insight)

    # -----------------------------------------------
    # Download Cleaned Dataset
    # -----------------------------------------------

    st.subheader("📥 Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )    

    # -----------------------------------------------
    # ML Prediction Module
    # -----------------------------------------------

    st.subheader("🤖 Auto ML Prediction")

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    if st.button("Train ML Model"):

        results = train_classification_model(
            df,
            target_column
        )

        st.success("Model trained successfully!")

        # Model Information
        st.write(
            f"✅ Model: {results['model_name']}"
        )

        st.write(
            f"🎯 Accuracy: {results['accuracy']:.2f}"
        )

        st.write(
            f"📊 Number of Features: {results['feature_count']}"
        )

        # -------------------------------------------
        # Confusion Matrix
        # -------------------------------------------

        st.subheader("📌 Confusion Matrix")

        cm_df = pd.DataFrame(
            results["confusion_matrix"]
        )

        st.dataframe(cm_df)
        
else:
    st.info("Please upload a CSV file.")

