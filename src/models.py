from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def train_models(X_train, y_train):
    """Trains LR, Decision Tree, and Random Forest models."""
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)

    rf = RandomForestRegressor(random_state=42, n_estimators=200)
    rf.fit(X_train, y_train)

    return lr, dt, rf

def evaluate_models(models, X_test, y_test):
    """Evaluates models and prints R2 Score & MSE."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = {'R2': r2, 'MSE': mse}
        print(f"{name}:\n  R2 Score: {r2:.3f}\n  MSE: {mse:.3f}\n")
    return results

def predict_new_student(models, study_hours, attendance, previous_perf):
    """Predict marks for a new student."""
    new_data = pd.DataFrame([[study_hours, attendance, previous_perf]],
                            columns=['Study_Hours', 'Attendance', 'Previous_Performance'])
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(new_data)[0]
    return predictions
