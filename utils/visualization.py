import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(df, column):

    fig, ax = plt.subplots()

    ax.hist(
        df[column].dropna(),
        bins=20
    )

    ax.set_title(f"Distribution of {column}")

    ax.set_xlabel(column)

    ax.set_ylabel("Frequency")

    return fig


def plot_correlation_heatmap(df):

    numerical_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    corr = numerical_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    return fig

def plot_boxplot(df, column):

    fig, ax = plt.subplots()

    ax.boxplot(df[column].dropna())

    ax.set_title(f"Boxplot of {column}")

    return fig

def plot_scatter(df, x_col, y_col):

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.scatter(
        df[x_col],
        df[y_col]
    )

    ax.set_xlabel(x_col)

    ax.set_ylabel(y_col)

    ax.set_title(f"{x_col} vs {y_col}")

    return fig