import pandas as pd
import time
import os       # <--- Add this line!
import joblib   # <--- Make sure this is here too!
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

# only run selected models if specified
def run_automl(filepath, target_column, selected_models=None):
    # 1. Load Data
    # sep=None allows Pandas to sniff the delimiter (comma, semicolon, etc.) automatically
    # engine='python' is required when using sep=None
    df = pd.read_csv(filepath, sep=None, engine='python')
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # If the target is categorical (strings), encode it to integers (0, 1, 2...)
    # This is required for XGBoost.
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # 2. Preprocessing Setup
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    # Pipeline for Numerical: Impute Missing -> Scale
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for Categorical: Impute Missing -> OneHot Encode
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
        if selected_models is not None and name not in selected_models:
            continue

        print(f"Training {name}...")
        
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', config['model'])])
        
        start_time = time.time()
        monitor = ResourceMonitor()
        
        with monitor:
            search = RandomizedSearchCV(clf, config['params'], n_iter=2, cv=2, n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
        
        duration = round(time.time() - start_time, 2)
        best_model = search.best_estimator_
        
        # STORE THE TRAINED MODEL OBJECT
        trained_models[name] = best_model 

        y_pred = best_model.predict(X_test)
        
        metrics = {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1 Score": round(f1_score(y_test, y_pred, average='weighted'), 4),
            "Precision": round(precision_score(y_test, y_pred, average='weighted'), 4),
            "Recall": round(recall_score(y_test, y_pred, average='weighted'), 4),
            "Training Time (s)": duration,
            "Max RAM (MB)": round(monitor.max_ram, 2),
            "Max CPU (%)": monitor.max_cpu,
            "Best Params": search.best_params_
        }
        results.append(metrics)

    # ... sorting logic ...
    results.sort(key=lambda x: (x['Accuracy'], -x['Training Time (s)']), reverse=True)

    # --- NEW: SAVE ALL MODELS ---
    # Get the absolute path to the 'backend/models' folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Loop through the dictionary of trained models and save EACH one
    for name, model in trained_models.items():
        # Clean the name (e.g., "Logistic Regression" -> "Logistic_Regression.pkl")
        safe_name = name.replace(" ", "_") + ".pkl"
        save_path = os.path.join(models_dir, safe_name)
        
        joblib.dump(model, save_path)
        print(f"Saved {name} to {save_path}")
    # ----------------------------

    return results