import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    
    try:
        df_drug = pd.read_csv("drug200.csv")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

label_encoder = LabelEncoder()

categorical_features = [feature for feature in df_drug.columns if df_drug[feature].dtypes == 'O']
for feature in categorical_features:
    df_drug[feature]=label_encoder.fit_transform(df_drug[feature])
    
X = df_drug.drop("Drug", axis=1)
y = df_drug["Drug"]

with mlflow.start_run():
    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X, y)

    kfold = KFold(random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

    print("  MEAN: %s" % cv_results.mean())
    print("  STD: %s" % cv_results.std())

    mlflow.log_param("X", X)
    mlflow.log_param("y", y)
    mlflow.log_metric("MEAN", cv_results.mean())
    mlflow.log_metric("STD", cv_results.std())

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetWineModel")
    else:
        mlflow.sklearn.log_model(model, "model")
