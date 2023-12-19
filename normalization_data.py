from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
import pandas as pd

def normalize_data(data, method):
    class_label_column = data.columns[-1]
    X = data.drop(columns=[class_label_column])

    if method == 1:
        scaler = MinMaxScaler()

    elif method == 2:
        scaler = StandardScaler()

    elif method == 3:
        scaler = RobustScaler()

    elif method == 4:
        scaler = Normalizer()

    elif method == 5:
        scaler = QuantileTransformer()

    elif method == 6:
        scaler = PowerTransformer()
    else:
        raise ValueError("Invalid method. Method should be an integer from 1 to 6.")

    normalized_data = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    normalized_data[class_label_column] = data[class_label_column]
    print_norm = f"\nSelected normalization method: {scaler}"
    return normalized_data, print_norm