# for generate insights
def generate_basic_insights(df):

    insights = []
    # Missing Values
    missing = df.isnull().sum()

    for column, value in missing.items():
        if value > 0:
            insights.append(
                f"Column '{column}' contains {value} missing values."
            )

    # duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:

        insights.append(
            f"Dataset contains {duplicates} duplicate rows."
        )

    # numerical Columns
    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for col in numerical_cols:

        mean_value = df[col].mean()

        insights.append(
            f"Average value of '{col}' is {mean_value:.2f}."
        )

    return insights
