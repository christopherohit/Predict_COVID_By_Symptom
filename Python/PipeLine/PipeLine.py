from itertools import tee
from pathlib import Path
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer , SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import numpy as np


def pipeline():
    
    data_dir = Path("D:\Lesson\Data Set\titannic")
    df_train_full = pd.read_csv(data_dir + "\traintitanic.csv")
    df_test = pd.read_csv(data_dir + "\test.csv")
    df_train_full.drop(columns=["Cabin"])
    df_test.drop(columns=["Cabin"]);
    df_train , df_val = train_test_split(df_train_full, test_size=0.1)
    X_train = df_train.copy()
    Y_train = X_train.pop("Survived")
    X_val = df_val.copy()
    Y_val = X_val.pop("Survived")


#  DataClean():
    cat_cols = ["Embarked", "Sex", "Pclass"]
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse= False)),
        ]
    )


#   DataProcessing():
    num_cols = ["Age", "Fare"]
    num_transformer = Pipeline(
        steps=[("imputer", KNNImputer(n_neighbors=5)),
        ("Scaler", RobustScaler)]
    )

    # Kết hợp hai bộ xử lý đặc trưng lại để có một bộ xử lý đặc trưng hoàn thiện. Lớp ColumnTransformer trong 
    # scikit-learn giúp 
    # kết hợp các transformers lại

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer , num_cols),
            ("cat", cat_transformer , cat_cols),
        ]
    )

    # Cuối cùng, ta kết hợp bộ xử lý đặc trưng preprocessor với một bộ phân loại đơn giản hay được sử dụng với 
    # dữ liệu dạng bảng 
    # là RandomForestClassifier để được một pipeline full_pp hoàn chỉnh bao gồm cả xử lý dữ liệu và mô hình. 
    # full_pp được fit với dữ liệu huấn luyện (X_train, y_train) sau đó được dùng để áp dụng lên dữ liệu kiểm 
    # định:

    #Full training pipeline

    full_pp = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
    )

    # training
    
    full_pp.fit(X_train,Y_train)

    # training metric

    y_train_pred = full_pp.predict(X_train)
    print(f"Accuracy score on train data: {accuracy_score(list(Y_train), list(y_train_pred)): .2f}")

    # validation metrics
    y_pred = full_pp.predict(X_val)
    print(f"Accuracy Score on validation data: {accuracy_score(list(Y_val), list(y_pred)): .2f}")

    preds = full_pp.predict(df_test)
    sample_submisson = pd.read_csv