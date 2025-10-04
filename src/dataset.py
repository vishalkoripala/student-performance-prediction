import pandas as pd
import numpy as np
import os

def create_dataset(num_students=500, seed=42):
    """
    Creates a synthetic dataset with realistic student features and marks.
    """
    np.random.seed(seed)
    study_hours = np.random.normal(6, 2, num_students)
    study_hours = np.clip(study_hours, 1, 10)

    attendance = np.random.normal(80, 10, num_students)
    attendance = np.clip(attendance, 50, 100)

    previous_performance = np.random.normal(70, 15, num_students)
    previous_performance = np.clip(previous_performance, 40, 100)

    marks = (0.4*study_hours*10 + 0.3*attendance + 0.3*previous_performance
             + np.random.normal(0, 5, num_students))

    df = pd.DataFrame({
        'Study_Hours': np.round(study_hours, 1),
        'Attendance': np.round(attendance, 1),
        'Previous_Performance': np.round(previous_performance, 1),
        'Marks': np.round(marks, 1)
    })

    # Save dataset
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/synthetic_student_data.csv', index=False)
    print("✅ Dataset created at data/synthetic_student_data.csv")
    return df
