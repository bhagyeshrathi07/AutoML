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
    """
    Defines all models. 
    We add class_weight='balanced' to force models to pay attention to minority classes (e.g. Fraud).
    """
    return {
        'Logistic Regression': {
            # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
            'model': LogisticRegression(solver='liblinear', class_weight='balanced'),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(class_weight='balanced'),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__criterion': ['gini', 'entropy']
            }
        },
        'SVM': {
            'model': SVC(probability=True, class_weight='balanced'),
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
            # XGBoost handles weights differently (scale_pos_weight), handled dynamically in the loop
            'model': XGBClassifier(eval_metric='logloss'),
            'params': {
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 5, 7]
            }
        }   
    }

def run_automl(filepath, target_column, selected_models=None):
    # 1. Load Data
    df = pd.read_csv(filepath, sep=None, engine='python')
    
    # --- CLEANING ---
    for col in df.columns:
        if 'id' in col.lower().split('_') or col.lower() == 'id':
             df = df.drop(columns=[col])
             print(f"🗑 Dropped ID column: {col}")

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=1, how='all')
    
    # Handle strings
    for col in list(df.select_dtypes(include=['object']).columns):
        if col == target_column: continue
        
        numeric_conversion = pd.to_numeric(df[col], errors='coerce')
        if numeric_conversion.notna().mean() > 0.8:
            df[col] = numeric_conversion
            continue

        try:
            pd.to_datetime(df[col], errors='raise')
            df = df.drop(columns=[col])
            continue
        except:
            pass

        if df[col].nunique() > 50 and (df[col].nunique() / len(df)) > 0.05:
            df = df.drop(columns=[col])
            print(f"💣 Dropped High-Cardinality column: {col}")
    # ----------------

    # --- NEW: STRATIFIED SAMPLING ---
    MAX_ROWS = 100000 
    
    if MAX_ROWS and len(df) > MAX_ROWS:
        print(f"⚠️ Dataset is huge ({len(df)} rows). Sampling down to {MAX_ROWS} (Stratified).")
        try:
            # Stratify ensures we keep the EXACT SAME ratio of Fraud vs Non-Fraud
            df, _ = train_test_split(df, train_size=MAX_ROWS, stratify=df[target_column], random_state=42)
        except:
            # Fallback to random if stratification fails (e.g. extremely rare classes)
            df = df.sample(n=MAX_ROWS, random_state=42)
    # --------------------------------

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # 2. Preprocessing
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    results = []
    configs = get_model_configs()
    trained_models = {} 

    # Calculate imbalance ratio for XGBoost
    # (Negatives / Positives)
    scale_pos_weight = 1
    if len(np.unique(y)) == 2:
        negatives = np.sum(y_train == 0)
        positives = np.sum(y_train == 1)
        if positives > 0:
            scale_pos_weight = negatives / positives

    # 4. Iterate Through Models
    for name, config in configs.items():
        if selected_models is not None and name not in selected_models:
            continue

        print(f"Training {name}...")
        
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', config['model'])])
        
        # DYNAMIC WEIGHT FOR XGBOOST
        if name == 'XGBoost' and len(np.unique(y)) == 2:
            clf.set_params(classifier__scale_pos_weight=scale_pos_weight)
            print(f"   -> Applied XGBoost scale_pos_weight: {round(scale_pos_weight, 2)}")

        start_time = time.time()
        monitor = ResourceMonitor()
        
        with monitor:
            search = RandomizedSearchCV(clf, config['params'], n_iter=2, cv=2, n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
        
        duration = round(time.time() - start_time, 2)
        best_model = search.best_estimator_
        trained_models[name] = best_model 

        y_pred = best_model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        cm_data = cm.tolist() 

        roc_data = []
        roc_auc_score = 0
        
        if len(np.unique(y)) == 2:
            try:
                y_prob = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc_score = auc(fpr, tpr)
                
                if len(fpr) < 20:
                    for i in range(len(fpr)):
                        roc_data.append({"x": round(fpr[i], 3), "y": round(tpr[i], 3)})
                else:
                    for i in range(0, len(fpr), 5): 
                        roc_data.append({"x": round(fpr[i], 3), "y": round(tpr[i], 3)})
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

    results.sort(key=lambda x: (x['Accuracy'], -x['Training Time (s)']), reverse=True)

    # Save Models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    for name, model in trained_models.items():
        safe_name = name.replace(" ", "_") + ".pkl"
        save_path = os.path.join(models_dir, safe_name)
        joblib.dump(model, save_path)
        print(f"Saved {name} to {save_path}")

    return results