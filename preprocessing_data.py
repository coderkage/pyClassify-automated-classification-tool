from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def preprocess_data(data):
    data = data.drop(columns=data.columns[0])

    imputer = SimpleImputer(strategy='most_frequent')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    class_label_column = data.columns[-1]
    X = data.drop(columns=[class_label_column])
    y = data[class_label_column]

    if y.dtype == 'O':
        label_encoder_y = LabelEncoder()
        y = label_encoder_y.fit_transform(y)

    for column in X.columns:
        if X[column].dtype == 'O' and not X[column].apply(lambda x: isinstance(x, (int, float))).all():
            label_encoder_X = LabelEncoder()
            X[column] = label_encoder_X.fit_transform(X[column])

    return pd.concat([X, pd.DataFrame(y, columns=[class_label_column])], axis=1)