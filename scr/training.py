import joblib

from preprocessing import preprocess_data

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve,precision_score,recall_score,f1_score)

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

import pandas as pd

import joblib

# Load Preprocessed Data
X_train, X_test, y_train, y_test = preprocess_data()

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)

# # for logistic regression
# X_train_final, y_train_final = smote.fit_resample(X_train_scaled,y_train)

#for random forest classifier
X_train_final, y_train_final = smote.fit_resample(X_train,y_train
)

# print("Before SMOTE:")
# print(y_train.value_counts())

# print("\nAfter SMOTE:")
# print(y_train_final.value_counts())

# # Model Training
# model = LogisticRegression(
#     random_state=42,
#     max_iter=1000
# )

# # Random Forest Classifier
# model=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
# model.fit(X_train_final, y_train_final)

# # Hyperparameter Tuning
# param_grid = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [10, 20, 30, None],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": ["sqrt", "log2"]
# }

# rf = RandomForestClassifier(
#     random_state=42,
#     n_jobs=-1
# )

# random_search = RandomizedSearchCV(
#     estimator=rf,
#     param_distributions=param_grid,
#     n_iter=10,          
#     cv=3,               
#     scoring="f1",
#     random_state=42,
#     n_jobs=-1,
#     verbose=2

# )

# random_search.fit(X_train_final, y_train_final)

# print("\nBest Parameters:")
# print(random_search.best_params_)

# print("\nBest Cross Validation Score:")
# print(random_search.best_score_)

# model = random_search.best_estimator_

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features="log2",
    random_state=42,
    n_jobs=1
)

model.fit(X_train_final, y_train_final)
model.set_params(n_jobs=1)

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    # Prediction
    # y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    threshold = 0.25
    y_pred = (y_prob >= threshold).astype(int)

    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nAccuracy:") 
    print(accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_score = roc_auc_score(y_test, y_prob)

    print("\nROC-AUC Score:")
    print(roc_score)

    # # ROC Curve
    # fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # plt.figure(figsize=(8, 6))

    # plt.plot(fpr, tpr, label=f"AUC = {roc_score:.3f}")

    # plt.plot([0, 1], [0, 1], linestyle="--")

    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")

    # plt.title("ROC Curve")

    # plt.legend()

    # plt.show()

## Logistic Regression
# evaluate_model(model, X_test_scaled, y_test)

## Random Forest Classifier
# evaluate_model(model, X_test, y_test)

# Feature Importance
def feature_importance(model, feature_names):

    model.set_params(n_jobs=1)
    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance:\n")
    print(feature_df)

    plt.figure(figsize=(10,6))

    plt.barh(
        feature_df["Feature"],
        feature_df["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()

    plt.show()

# feature_importance(model,X_train.columns)

# save model
joblib.dump(model, "models/randomforest_model.pkl")

#save model
joblib.dump(model, "models/random_forest_model.pkl")

#save scaler
joblib.dump(scaler, "models/scaler.pkl")

#save threshold
joblib.dump(0.25, "models/threshold.pkl")

print("\nModel Saved Successfully.")