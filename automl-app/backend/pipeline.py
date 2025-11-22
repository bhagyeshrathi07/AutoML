import pandas as pd
import numpy as np
import time
import os
import joblib
import re
import random

# Scikit-Learn & XGBoost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_curve, auc, confusion_matrix, 
                             r2_score, mean_squared_error, mean_absolute_error)

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from monitor import ResourceMonitor

def clean_column_names(df):
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df.columns]
    return df

def get_configs(task_type, n_rows):
    """
    Returns model configurations.
    Dynamic Logic: If dataset is large (>10k rows), swaps slow SVM/KNN for fast approximations.
    """
    is_large_data = n_rows > 10000
    
    print(f"⚡ Model Selection Mode: {'High-Performance (Large Data)' if is_large_data else 'High-Accuracy (Small Data)'}")

    if task_type == 'Classification':
        # FAST SVM: SGDClassifier (O(n)) vs SLOW SVM: SVC (O(n^3))
        svm_config = {
            'model': SGDClassifier(loss='hinge', class_weight='balanced', random_state=42),
            'params': {'classifier__alpha': [0.0001, 0.001, 0.01], 'classifier__penalty': ['l2', 'elasticnet']}
        } if is_large_data else {
            'model': SVC(probability=True, class_weight='balanced', random_state=42),
            'params': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}
        }

        return {
            'Logistic Regression': {'model': LogisticRegression(solver='liblinear', class_weight='balanced'), 'params': {'classifier__C': [0.1, 1, 10]}},
            'Random Forest': {'model': RandomForestClassifier(class_weight='balanced'), 'params': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [10, 20]}},
            'SVM': svm_config,
            'KNN': {'model': KNeighborsClassifier(), 'params': {'classifier__n_neighbors': [3, 5, 7]}},
            'XGBoost': {'model': XGBClassifier(eval_metric='logloss'), 'params': {'classifier__learning_rate': [0.01, 0.1], 'classifier__n_estimators': [50, 100]}}
        }
    else: # Regression
        # FAST SVR: LinearSVR (O(n)) vs SLOW SVR: SVR (O(n^3))
        # SGDRegressor is even faster for massive data
        svr_config = {
            'model': SGDRegressor(random_state=42),
            'params': {'classifier__alpha': [0.0001, 0.001], 'classifier__penalty': ['l2', 'elasticnet']}
        } if is_large_data else {
            'model': SVR(),
            'params': {'classifier__C': [0.1, 1], 'classifier__kernel': ['linear', 'rbf']}
        }

        return {
            'Linear Regression': {'model': LinearRegression(), 'params': {}},
            'Ridge': {'model': Ridge(), 'params': {'classifier__alpha': [0.1, 1.0, 10.0]}},
            'Lasso': {'model': Lasso(), 'params': {'classifier__alpha': [0.1, 1.0, 5.0]}},
            'Decision Tree': {'model': DecisionTreeRegressor(), 'params': {'classifier__max_depth': [5, 10, 20]}},
            'Random Forest': {'model': RandomForestRegressor(), 'params': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [10, 20]}},
            'KNN': {'model': KNeighborsRegressor(), 'params': {'classifier__n_neighbors': [3, 5, 7]}},
            'SVM': svr_config, # Renamed from SVR to keep key consistent if desired, or use SVR
            'XGBoost': {'model': XGBRegressor(), 'params': {'classifier__learning_rate': [0.01, 0.1], 'classifier__n_estimators': [50, 100]}}
        }

def run_automl(filepath, target_column, selected_models=None, callback=None):
    if callback: callback(5, "📂 Loading dataset...")
    df = pd.read_csv(filepath)
    df = clean_column_names(df)
    
    # --- 1. ROBUST DATA CLEANING ---
    if callback: callback(10, "🧹 Cleaning data...")
    
    for col in df.columns:
        if 'id' in col.lower().split('_') or col.lower() == 'id':
             df = df.drop(columns=[col])

    df = df.dropna(axis=1, how='all')
    df = df.dropna(subset=[target_column])

    # High Cardinality Drop
    for col in df.select_dtypes(include=['object']).columns:
        if col == target_column: continue
        if df[col].nunique() > 50 and (df[col].nunique() / len(df)) > 0.9:
            df = df.drop(columns=[col])

    # --- 2. TASK DETECTION ---
    y = df[target_column]
    task_type = "Classification"
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() > 20:
            task_type = "Regression"
            
    if callback: callback(15, f"🔍 Task: {task_type} | Rows: {len(df)}")

    # --- 3. PREPROCESSING ---
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if task_type == "Classification" and (y.dtype == 'object' or isinstance(y.iloc[0], str)):
        le = LabelEncoder()
        y = le.fit_transform(y)

    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, X.select_dtypes(include=['number']).columns), 
        ('cat', cat_transformer, X.select_dtypes(exclude=['number']).columns)
    ])

    if task_type == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. TRAINING LOOP ---
    # PASS n_rows to config to trigger smart switching
    configs = get_configs(task_type, len(X_train))
    
    results = []
    trained_models = {}
    
    valid_models = configs.keys()
    # Fuzzy matching for model selection (e.g., "SVM" matches "SVM (Linear)")
    models_to_run = []
    if selected_models:
        for m in selected_models:
            if m in configs:
                models_to_run.append(m)
            elif m == "SVM" and "SVM" in configs: # Handle naming variations
                models_to_run.append("SVM")
    else:
        models_to_run = list(valid_models)
    
    total_models = len(models_to_run)
    current_model_idx = 0

    for name in models_to_run:
        config = configs[name]
        current_model_idx += 1
        progress_pct = 20 + int((current_model_idx / total_models) * 70)
        if callback: callback(progress_pct, f"🏋️ Training {name}...")

        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', config['model'])])
        
        start_time = time.time()
        monitor = ResourceMonitor()
        
        with monitor:
            # Reduce iterations for large data to ensure speed
            n_iter = 2 if len(X_train) > 10000 else 5
            search = RandomizedSearchCV(clf, config['params'], n_iter=n_iter, cv=3, n_jobs=-1, random_state=42)
            try:
                search.fit(X_train, y_train)
            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue

        duration = round(time.time() - start_time, 2)
        best_model = search.best_estimator_
        trained_models[name] = best_model 
        y_pred = best_model.predict(X_test)

        metrics = {
            "Model": name,
            "Task Type": task_type,
            "Training Time (s)": duration,
            "Max RAM (MB)": round(monitor.max_ram, 2),
            "Max CPU (%)": monitor.max_cpu,
            "Best Params": search.best_params_
        }

        if task_type == "Classification":
            metrics.update({
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 4),
                "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist()
            })
            if len(np.unique(y)) == 2:
                try:
                    # Check if model supports predict_proba (SGD with hinge loss does NOT)
                    if hasattr(best_model.named_steps['classifier'], 'predict_proba'):
                        y_prob = best_model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        metrics["AUC"] = round(auc(fpr, tpr), 4)
                        metrics["ROCData"] = [{"x": round(f,3), "y": round(t,3)} for f,t in zip(fpr[::5], tpr[::5])]
                    else:
                        metrics["AUC"] = "N/A (Linear SVM)"
                        metrics["ROCData"] = []
                except: pass
        else:
            metrics.update({
                "R2 Score": round(r2_score(y_test, y_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                "MAE": round(mean_absolute_error(y_test, y_pred), 4)
            })
            
            indices = np.arange(len(y_test))
            np.random.shuffle(indices)
            sample_indices = indices[:200]
            
            scatter_data = []
            y_test_arr = np.array(y_test)
            for i in sample_indices:
                scatter_data.append({
                    "actual": float(round(y_test_arr[i], 2)),
                    "predicted": float(round(y_pred[i], 2))
                })
            metrics["ScatterData"] = scatter_data

        results.append(metrics)

    if task_type == "Classification":
        results.sort(key=lambda x: x.get('Accuracy', 0), reverse=True)
    else:
        results.sort(key=lambda x: x.get('R2 Score', -float('inf')), reverse=True)

    models_dir = 'models'
    for name, model in trained_models.items():
        joblib.dump(model, os.path.join(models_dir, name.replace(" ", "_") + ".pkl"))

    return results