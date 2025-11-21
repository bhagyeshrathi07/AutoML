import pandas as pd
import time
import os
import joblib
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from monitor import ResourceMonitor

def get_model_configs():
    """Defines all models and their hyperparameter grids."""
    return {
        'Logistic Regression': {
            'model': LogisticRegression(solver='liblinear'),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            }
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'classifier__n_neighbors': [3, 5, 7, 9],
                'classifier__weights': ['uniform', 'distance']
            }
        },
        'XGBoost': {
            'model': XGBClassifier(eval_metric='logloss'),
            'params': {
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 5, 7]
            }
        }   
    }

def run_automl(filepath, target_column, selected_models=None):
    # 1. Load Data (Smart Parsing)
    df = pd.read_csv(filepath, sep=None, engine='python')
    
    # --- ROBUST CLEANING LOGIC ---
    # A. Drop 'id' columns (case-insensitive) to prevent cheating/overfitting
    if 'id' in df.columns.str.lower():
        col_to_drop = df.columns[df.columns.str.lower() == 'id'][0]
        df = df.drop(columns=[col_to_drop])
        print(f"Dropped ID column: {col_to_drop}")

    # B. Drop 'Unnamed' columns (common in Kaggle datasets)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # C. Drop columns that are completely empty (All NaN)
    df = df.dropna(axis=1, how='all')
    # -----------------------------

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # If the target is categorical (strings), encode it to integers (0, 1, 2...)
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # 2. Preprocessing Setup
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    configs = get_model_configs()
    
    # Dictionary to temporarily hold the actual trained model objects
    trained_models = {} 

    # 4. Iterate Through Models
    for name, config in configs.items():
        # Skip model if not selected by user
        if selected_models is not None and name not in selected_models:
            continue

        print(f"Training {name}...")
        
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', config['model'])])
        
        start_time = time.time()
        monitor = ResourceMonitor()
        
        with monitor:
            # Faster search settings (iter=2, cv=2) for responsiveness
            search = RandomizedSearchCV(clf, config['params'], n_iter=2, cv=3, n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
        
        duration = round(time.time() - start_time, 2)
        best_model = search.best_estimator_
        
        # Store the trained model object for saving later
        trained_models[name] = best_model 

        y_pred = best_model.predict(X_test)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_data = cm.tolist() # Convert to list for JSON serialization

        # 2. ROC Curve Data & AUC (Only for binary classification)
        roc_data = []
        roc_auc_score = 0
        
        if len(np.unique(y)) == 2:  # Check if binary
            try:
                y_prob = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc_score = auc(fpr, tpr)
                
                # --- ROC Visual Fix ---
                # If the curve is perfect (very few points), keep ALL of them.
                # Otherwise, downsample to save bandwidth.
                if len(fpr) < 20:
                    for i in range(len(fpr)):
                        roc_data.append({"x": round(fpr[i], 3), "y": round(tpr[i], 3)})
                else:
                    for i in range(0, len(fpr), 5): 
                        roc_data.append({"x": round(fpr[i], 3), "y": round(tpr[i], 3)})
                    # Always append the last point
                    roc_data.append({"x": round(fpr[-1], 3), "y": round(tpr[-1], 3)})
            except:
                roc_data = None
        
        metrics = {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 4),
            "Precision": round(precision_score(y_test, y_pred, average='weighted'), 4),
            "Recall": round(recall_score(y_test, y_pred, average='weighted'), 4),
            "Training Time (s)": duration,
            "Max RAM (MB)": round(monitor.max_ram, 2),
            "Max CPU (%)": monitor.max_cpu,
            "Best Params": search.best_params_,
            "AUC": round(roc_auc_score, 4) if roc_data else None,
            "ConfusionMatrix": cm_data,  
            "ROCData": roc_data          
        }
        results.append(metrics)

    # Sort results by Accuracy (Desc) then Time (Asc)
    results.sort(key=lambda x: (x['Accuracy'], -x['Training Time (s)']), reverse=True)

    # --- SAVE ALL TRAINED MODELS ---
    # Get absolute path to ensure it works regardless of where python is run
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    for name, model in trained_models.items():
        # Clean the name (e.g., "Logistic Regression" -> "Logistic_Regression.pkl")
        safe_name = name.replace(" ", "_") + ".pkl"
        save_path = os.path.join(models_dir, safe_name)
        
        joblib.dump(model, save_path)
        print(f"Saved {name} to {save_path}")
    # -------------------------------

    return results