from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, LeaveOneOut, KFold, ShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fpdf import FPDF

def cross_validation_and_evaluation(model, data, cv_type):
    class_label_column = data.columns[-1]

    X = data.drop(columns=[class_label_column])
    y = data[class_label_column]

    if cv_type == 1:
        num_folds = int(input("Enter the number of folds: "))
        cv = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        cv_name = 'k-fold'
        print(f"Performing {num_folds}-fold k-fold cross-validation...\n")

    elif cv_type == 2:
        num_folds = int(input("Enter the number of folds: "))
        cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        cv_name = 'stratified k-fold'
        print(f"Performing {num_folds}-fold stratified k-fold cross-validation...\n")

    elif cv_type == 3:
        cv = LeaveOneOut()
        cv_name = 'LOOCV'
        print(f"Performing Leave Out One Cross Validation...\n")

    elif cv_type == 4:
        num_folds = int(input("Enter the number of folds: "))
        cv = ShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=42)
        cv_name = 'Shuffle Split'
        print(f"Performing {num_folds}-fold Shuffle Split cross-validation...\n")

    elif cv_type == 5:
        num_folds = int(input("Enter the number of folds: "))
        cv = StratifiedShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=42)
        cv_name = 'Stratified Shuffle Split'
        print(f"Performing {num_folds}-fold Stratified Shuffle Split cross-validation...\n")

    elif cv_type == 6:
        num_folds = int(input("Enter the number of folds: "))
        cv = TimeSeriesSplit(n_splits=num_folds)
        cv_name = 'Time Series Split'
        print(f"Performing {num_folds}-fold Time Series Split cross-validation...\n")

    else:
        raise ValueError("Invalid method. Method should be an integer from 1 to 6.")

    scores = cross_val_score(model, X, y, cv=cv)

    plt.plot(np.arange(1, len(scores) + 1), scores, marker='o')
    plt.title(f'Model Index vs Accuracy ({num_folds}-fold {cv_name})')
    plt.xlabel('Model Index')
    plt.ylabel('Accuracy')
    plt.savefig("accuracy_graph.png")
    plt.show()

    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold}: Accuracy = {score * 100:.2f}%")

    best_fold_index = np.argmax(scores)
    print(f"\nTraining on the best fold ({best_fold_index + 1})...")

    if cv_type != 3:
        train_index, test_index = list(cv.split(X, y))[best_fold_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    else:
        test_index = best_fold_index
        train_index = np.delete(np.arange(len(X)), test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    trained_model = model.fit(X_train, y_train)

    y_pred = trained_model.predict(X_test)
    a = accuracy_score(y_test, y_pred)
    print("\nEvaluation on the test data of the best fold:")
    print("Accuracy:", a)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix - Training Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("train_heatmap.png")
    plt.show()

    print_cv = f"\nCross Validation method: {cv_name}"
    print_acc = f"\nAccuracy on training data set: {a}"
    return trained_model, print_cv, print_acc
