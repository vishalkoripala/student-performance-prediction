import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from dataset import create_dataset
from models import train_models, evaluate_models, predict_new_student

# Step 1: Create Dataset
df = create_dataset()

# Step 2: Prepare Features & Target
X = df[['Study_Hours', 'Attendance', 'Previous_Performance']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Models
lr, dt, rf = train_models(X_train, y_train)
models = {'Linear Regression': lr, 'Decision Tree': dt, 'Random Forest': rf}

# Step 4: Evaluate Models
results = evaluate_models(models, X_test, y_test)

# Step 5: Visualize R2 Scores
r2_scores = [v['R2'] for v in results.values()]
plt.figure(figsize=(7,5))
sns.barplot(x=list(results.keys()), y=r2_scores)
plt.title("Model Accuracy Comparison (R² Score)")
plt.ylabel("R² Score")

if not os.path.exists('reports'):
    os.makedirs('reports')
plt.savefig('reports/performance_report.png')
plt.show()

# Step 6: Predict New Student
preds = predict_new_student(models, study_hours=7, attendance=85, previous_perf=72)
print("Predicted Marks for new student:", preds)
