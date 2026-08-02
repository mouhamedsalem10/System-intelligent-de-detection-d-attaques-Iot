import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

X_train_sc = np.load("data/X_train_scaled.npy")
y_train    = np.load("data/y_train.npy")
X_eval_sc  = np.load("data/X_eval_scaled.npy")
y_eval     = np.load("data/y_eval.npy")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_sc, y_train, test_size=0.2, random_state=42, stratify=y_train
)

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
params = {
    'objective': 'binary:logistic', 'eval_metric': ['logloss','auc'],
    'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'scale_pos_weight': scale_pos, 'seed': 42
}

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval   = xgb.DMatrix(X_val, label=y_val)
deval  = xgb.DMatrix(X_eval_sc, label=y_eval)

model = xgb.train(params, dtrain, num_boost_round=150,
                   evals=[(dtrain,'train'),(dval,'val')],
                   early_stopping_rounds=15, verbose_eval=False)

probas = model.predict(deval)
y_pred = (probas >= 0.5).astype(int)

print(f"BASELINE (features brutes seules, sans embeddings) :")
print(f"  Accuracy : {accuracy_score(y_eval,y_pred)*100:.2f}%")
print(f"  F1-Score : {f1_score(y_eval,y_pred):.4f}")
print(f"  AUC-ROC  : {roc_auc_score(y_eval,probas):.4f}")