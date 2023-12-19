from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, mutual_info_classif, f_regression, mutual_info_regression
import pandas as pd
import matplotlib.pyplot as plt

def select_features(data, method):
    class_label_column = data.columns[-1]
    X = data.drop(columns=[class_label_column])

    if method == 5:
        num_features = int(input("Enter the percentile(1-100): "))
    elif method == 1 or method == 2 or method == 3 or method == 4:
        num_features = int(input("Enter the no. of features: "))
    else:
        raise ValueError("Invalid method. Method should be an integer from 1 to 5.")

    if method == 1:
        selector = SelectKBest(f_classif, k=num_features)
        s = "SelectKBest with f_classif"
    elif method == 2:
        selector = SelectKBest(mutual_info_classif, k=num_features)
        s = "SelectKBest with mutual_info_classif"
    elif method == 3:
        selector = SelectKBest(f_regression, k=num_features)
        s = "SelectKBest with f_regression"
    elif method == 4:
        selector = SelectKBest(mutual_info_regression, k=num_features)
        s = "SelectKBest with mutual_info_regression"
    elif method == 5:
        selector = SelectPercentile(f_classif, percentile=num_features)
        s = "SelectPercentile with f_classif"
    else:
        raise ValueError("Invalid method. Method should be an integer from 1 to 5.")
    
    selector.fit(X, data[class_label_column])
    selected_feature_indices = selector.get_support(indices=True)
    selected_features_data = X.iloc[:, selected_feature_indices]
    selected_features_data[class_label_column] = data[class_label_column]

    print("Selected Features:")
    print(selected_features_data.columns[:-1])
    print_feature = f"\nSelected feature selection method: {s}"
    print_selected = f"\nSelected features: {selected_features_data.columns[:-1]}"
    return selected_features_data, print_feature, print_selected