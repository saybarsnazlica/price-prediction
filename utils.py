import re

import nltk
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.preprocessing import LabelBinarizer


def rmsle(y_true, y_pred):
    """Calculate root mean squared log error

    Args:
        y_true : True labels
        y_pred : Predicted labels

    Returns:
        float: Root mean squared log error value. squared=False in mean_squared_log_error() returns root mean squared log error
    """
    return mean_squared_log_error(y_true, y_pred, squared=False)


def calc_performance_metrics(model, X_train, X_valid, y_train, y_valid):
    """Calculate performance metrics on training and validation data

    Args:
        model : Machine learning model
        X_train : Train features
        X_valid : Validation features
        y_train : Train labels
        y_valid : Validation labels

    Returns:
        Dict : Metrics dictionary
    """
    train_preds = np.expm1(model.predict(X_train))
    val_preds = np.expm1(model.predict(X_valid))

    metrics = {
        "Training RMSLE": rmsle(y_train, train_preds),
        "Validation RMSLE": rmsle(y_valid, val_preds),
        "Training MAE": mean_absolute_error(y_train, train_preds),
        "Validation MAE": mean_absolute_error(y_valid, val_preds),
    }
    return metrics


def process_data(df):
    # Split category_name column into three columns
    df[["first_category", "second_category", "third_category"]] = df[
        "category_name"
    ].str.split("/", 2, expand=True)

    # Deleting unneeded column
    df.drop("category_name", axis=1, inplace=True)

    # Adding new features indicating missing data
    df["item_description"] = df["item_description"].replace(
        {"No description yet": np.nan}
    )
    df["category_missing"] = df["first_category"].isna()
    df["brand_missing"] = df["brand_name"].isna()
    df["description_missing"] = df["item_description"].isna()

    # Replacing NaN values with "missing"
    for i in [
        "brand_name",
        "first_category",
        "second_category",
        "third_category",
        "item_description",
    ]:
        df[i] = df[i].fillna("missing")

    # Adding features indicating that there was a price tag in item description or name
    df["price_in_description"] = df["item_description"].str.contains("\[rm\]")
    df["price_in_name"] = df["name"].str.contains("\[rm\]")

    # Adding a new column with description text length devided into 5 character intervals
    df["descr_len"] = df["item_description"].str.len()
    df["descr_len"] = pd.cut(df["descr_len"], np.arange(0, 1055, 5), right=False)
    df["descr_len"] = df["descr_len"].astype("string")

    stop_words = set(nltk.corpus.stopwords.words("english"))

    # Processing name and description columns
    for column in ["item_description", "brand_name", "name"]:
        processed_column = []
        for text_row in df[column]:
            text_row = text_row.replace("[rm]", "")
            text_row = re.sub("[^A-Za-z0-9]+", " ", text_row)
            if column != "brand_name":
                text_row = " ".join(
                    word for word in text_row.lower().split() if word not in stop_words
                )
            processed_column.append(text_row.strip())
        df[column] = processed_column

    # Processing category columns
    for column in ["first_category", "second_category", "third_category"]:
        processed_column = []
        for text_row in df[column]:
            text_row = text_row.replace(" ", "")
            text_row = text_row.replace("&", "_")
            text_row = re.sub("[^A-Za-z0-9_]+", " ", text_row)
            processed_column.append(text_row.lower().strip())
        df[column] = processed_column

    return df


def transform_train_valid_data(X_train, X_valid):
    """Transform data training and validation data

    Args:
        X_train : Training features
        X_valid : Validation features

    Returns:
        X_train_stack
        X_valid_stack
        label_binarizers
        count_vectorizers
    """
    cat_features = [
        "item_condition_id",
        "first_category",
        "second_category",
        "third_category",
        "shipping",
        "brand_name",
        "description_missing",
        "price_in_name",
        "price_in_description",
        "brand_missing",
        "category_missing",
        "descr_len",
    ]

    label_binarizers = []
    binarized_columns = []
    count_vectorizers = []
    vectorized_columns = []

    for column in cat_features:
        binarizer = LabelBinarizer(sparse_output=True)
        binarized_column = binarizer.fit_transform(X_train[column])
        label_binarizers.append(binarizer)
        binarized_columns.append(binarized_column)

    vectorizer = CountVectorizer(min_df=7, max_features=20000)
    vectorized_column = vectorizer.fit_transform(X_train["name"])
    count_vectorizers.append(vectorizer)
    vectorized_columns.append(vectorized_column)

    vectorizer = CountVectorizer(min_df=15, ngram_range=(1, 2), max_features=40000)
    vectorized_column = vectorizer.fit_transform(X_train["item_description"])
    count_vectorizers.append(vectorizer)
    vectorized_columns.append(vectorized_column)

    vectorizer = CountVectorizer(min_df=30, ngram_range=(3, 3), max_features=5000)
    vectorized_column = vectorizer.fit_transform(X_train["item_description"])
    count_vectorizers.append(vectorizer)
    vectorized_columns.append(vectorized_column)

    X_train_stack = hstack(
        (
            binarized_columns[0],
            binarized_columns[1],
            binarized_columns[2],
            binarized_columns[3],
            binarized_columns[4],
            binarized_columns[5],
            binarized_columns[6],
            binarized_columns[7],
            binarized_columns[8],
            binarized_columns[9],
            binarized_columns[10],
            binarized_columns[11],
            vectorized_columns[0],
            vectorized_columns[1],
            vectorized_columns[2],
        )
    ).tocsr()

    X_valid_stack = transform_test_data(X_valid, label_binarizers, count_vectorizers)

    return (
        X_train_stack,
        X_valid_stack,
        label_binarizers,
        count_vectorizers,
    )


def transform_test_data(df, label_binarizers, count_vectorizers):
    """Transform data with binarization and vectorization

    Args:
        df : Input data
        label_binarizers : Binarization object
        count_vectorizers : Vectorization object

    Returns:
        csr matrix : Stacked X_test
    """

    cat_features = [
        "item_condition_id",
        "first_category",
        "second_category",
        "third_category",
        "shipping",
        "brand_name",
        "description_missing",
        "price_in_name",
        "price_in_description",
        "brand_missing",
        "category_missing",
        "descr_len",
    ]

    binarized_columns = []
    vectorized_columns = []

    for num, column in enumerate(cat_features):
        binarizer = label_binarizers[num]
        binarized_column = binarizer.transform(df[column])
        binarized_columns.append(binarized_column)

    vectorizer = count_vectorizers[0]
    vectorized_column = vectorizer.transform(df["name"])
    vectorized_columns.append(vectorized_column)

    vectorizer = count_vectorizers[1]
    vectorized_column = vectorizer.transform(df["item_description"])
    vectorized_columns.append(vectorized_column)

    vectorizer = count_vectorizers[2]
    vectorized_column = vectorizer.transform(df["item_description"])
    vectorized_columns.append(vectorized_column)

    X_test_stack = hstack(
        (
            binarized_columns[0],
            binarized_columns[1],
            binarized_columns[2],
            binarized_columns[3],
            binarized_columns[4],
            binarized_columns[5],
            binarized_columns[6],
            binarized_columns[7],
            binarized_columns[8],
            binarized_columns[9],
            binarized_columns[10],
            binarized_columns[11],
            vectorized_columns[0],
            vectorized_columns[1],
            vectorized_columns[2],
        )
    ).tocsr()

    return X_test_stack
