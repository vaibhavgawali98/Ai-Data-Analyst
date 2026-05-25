import pandas as pd


def get_missing_values(df):

    missing_values = df.isnull().sum()

    missing_percentage = (
        df.isnull().sum() / len(df)
    ) * 100

    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values,
        "Percentage": missing_percentage.values
    })

    return missing_df


def remove_duplicates(df):

    df = df.drop_duplicates()

    return df


# ------------------------------------------------
# Fill Missing Values
# ------------------------------------------------

def fill_missing_values(df, column, method):

    if method == "Mean":

        df[column] = df[column].fillna(
            df[column].mean()
        )

    elif method == "Median":

        df[column] = df[column].fillna(
            df[column].median()
        )

    elif method == "Mode":

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

    return df