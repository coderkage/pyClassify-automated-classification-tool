import pandas as pd
from sklearn.model_selection import train_test_split
from fpdf import FPDF
from datetime import datetime

from preprocessing_data import preprocess_data
from normalization_data import normalize_data
from feature_selection import select_features
from classification_model import classifier_model
from cross_validation import cross_validation_and_evaluation
from blind_predict import predict_on_blind_data

file_path = input("Enter the file path of your dataset: ")

data = pd.read_csv(file_path)
data_set = f"\nSelected data set: {file_path}"
model_data, blind_data = train_test_split(data, test_size=0.1, random_state=42)
    
preprocessed_data = preprocess_data(model_data)
pr_blind_data=preprocess_data(blind_data)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(" 1. MinMaxScaler\n 2. StandardScaler\n 3. RobustScaler\n 4. Normalizer\n 5. QuantileTransformer\n 6. PowerTransformer\n")
norm_method = int(input("--> Choose the normalization method (1-6): "))

normalized_data, print_norm= normalize_data(preprocessed_data, norm_method)

# nan_columns = normalized_data.columns[normalized_data.isna().any()].tolist()
# print("Columns with NaN values:", nan_columns)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(" 1. SelectKBest with f_classif\n 2. SelectKBest with mutual_info_classif\n 3. SelectKBest with f_regression\n 4. SelectKBest with mutual_info_regression\n 5. SelectPercentile with f_classif\n")
feature_method = int(input("--> Choose the feature selection method (1-5): "))
selected_features_data, print_feature, print_selected = select_features(normalized_data, feature_method)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(" 1. Random Forest\n 2. Logistic Regression\n 3. Gradient Boosting\n 4. NaiveBayes\n 5. KNN\n 6. Decision Tree\n 7. AdaBoost\n 8. Bagging")
class_method = int(input("--> Choose the classification model (1-8): "))
model, print_c = classifier_model(class_method)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(" 1. K-Fold\n 2. Stratified K-fold\n 3. LOOCV\n 4. Shuffle Split\n 5. Stratified Shuffle Split")
crossval_method = int(input("--> Choose the classification model (1-5): "))
trained_model, cv_name, acc = cross_validation_and_evaluation(model, selected_features_data, crossval_method)

class_label_column_bl = pr_blind_data.columns[-1]
selected_features_blind = pr_blind_data[selected_features_data.columns[:-1]]
selected_features_blind[class_label_column_bl] = pr_blind_data[class_label_column_bl]

accuracy_on_blind_data = predict_on_blind_data(trained_model, selected_features_blind)
blind_accuracy = f"\nAccuracy on blind dataset: {accuracy_on_blind_data}"
####################################################################################################################################################

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="pyClassify - Automated ML Tool for Classification", ln=True, align='C')
pdf.cell(200, 10, txt="[Project Simulation Results]", ln=True, align='C')
pdf.ln(5)
pdf.multi_cell(200, 10, txt=data_set, align='L')
pdf.multi_cell(200, 10, txt=print_norm, align='L')
pdf.multi_cell(200, 10, txt=print_feature, align='L')
pdf.multi_cell(200, 10, txt=print_selected, align='L')
pdf.multi_cell(200, 10, txt=print_c, align='L')
pdf.multi_cell(200, 10, txt=cv_name, align='L')
pdf.cell(200, 10, txt=acc, ln=True, align='L')
pdf.cell(200, 10, txt=blind_accuracy, ln=True, align='L')
pdf.add_page()
pdf.image("accuracy_graph.png", x=10, y=10, w=190)
pdf.add_page()
pdf.image("train_heatmap.png", x=10, y=10, w=190)
pdf.add_page()
pdf.image("blind_heatmap.png", x=10, y=10, w=190)


pdf.output(f"report_{timestamp}.pdf")

print(f"Report PDF file created successfully!")