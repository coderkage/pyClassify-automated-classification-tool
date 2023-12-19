from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut, KFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def predict_on_blind_data(trained_model, blind_data):
    class_label_column = blind_data.columns[-1]

    X_blind = blind_data.drop(columns=[class_label_column])
    y_blind = blind_data[class_label_column]

    # Use the trained model to predict on the blind dataset
    y_pred_blind = trained_model.predict(X_blind)

    # Calculate accuracy on the blind dataset
    accuracy_blind = accuracy_score(y_blind, y_pred_blind)

    # Display results
    print("\nEvaluation on the blind dataset:")
    print("Accuracy:", accuracy_blind)
    print("\nClassification Report:\n", classification_report(y_blind, y_pred_blind))
    print("Confusion Matrix:\n", confusion_matrix(y_blind, y_pred_blind))

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_blind, y_pred_blind)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_blind), yticklabels=np.unique(y_blind))
    plt.title('Confusion Matrix - Blind Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("blind_heatmap.png")
    plt.show()

    return accuracy_blind
