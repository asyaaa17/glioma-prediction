from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from joblib import dump

def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_scaled, y_train)
    dummy_predictions = dummy.predict(X_test_scaled)

    print_metrics(y_test, dummy_predictions, 'Dummy Classifier')

    
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)
    
    print_metrics(y_test, lr_predictions, 'Logistic Regression')
    

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_scaled, y_train)
    rf_predictions = rf_model.predict(X_test_scaled)
    
    print_metrics(y_test, rf_predictions, 'Random Forest')
    dump(rf_model, 'random_forest_model.joblib')

    
    mlp_model = MLPClassifier()
    mlp_model.fit(X_train_scaled, y_train)
    mlp_predictions = mlp_model.predict(X_test_scaled)
    
    print_metrics(y_test, mlp_predictions, 'MLP Classifier')

    


def print_metrics(y_true, y_pred, model_name):
    print(f'{model_name} - F1 Score: {f1_score(y_true, y_pred)}')
    print(f'{model_name} - Accuracy: {accuracy_score(y_true, y_pred)}')
    print(f'{model_name} - Recall: {recall_score(y_true, y_pred)}')
    print(f'{model_name} - Precision: {precision_score(y_true, y_pred)}\n')