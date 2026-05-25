import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)


def train_classification_model(df, target_column):

    # Drop Missing Values
    df = df.dropna()

    # Features and Target
    X = df.drop(columns=[target_column])

    y = df[target_column]

    # Convert categorical columns
    X = pd.get_dummies(X)

    # Encode target column
    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train Model
    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "model_name": "Random Forest Classifier",
        "feature_count": X.shape[1],
        "confusion_matrix": cm
    }