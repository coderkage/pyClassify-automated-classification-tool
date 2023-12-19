from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

def classifier_model(method):
    if method == 1:
        classifier = RandomForestClassifier()

    elif method == 2:
        classifier = LogisticRegression()

    elif method == 3:
        classifier = GradientBoostingClassifier()

    elif method == 4:
        classifier = GaussianNB()

    elif method == 5:
        classifier = KNeighborsClassifier()

    elif method == 6:
        classifier = DecisionTreeClassifier()

    elif method == 7:
        classifier = AdaBoostClassifier()

    elif method == 8:
        classifier = BaggingClassifier()

    else:
        raise ValueError("Invalid method. Method should be an integer from 1 to 8.")

    print_c = f"\nSelected classifier model: {classifier}"
    return classifier, print_c