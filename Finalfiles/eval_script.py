import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor

# Set seed
np.random.seed(123)

# Import training data
trn = pd.read_csv('CW1_train.csv')
X_tst = pd.read_csv('CW1_test.csv') # This does not include true outcomes (obviously)

# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']

# One-hot encode categorical variables
trn = pd.get_dummies(trn, columns=categorical_cols, drop_first=True)
X_tst = pd.get_dummies(X_tst, columns=categorical_cols, drop_first=True)

# Train your model (using a simple LM here as an example)
selected_features = ['depth', 'a1', 'a3', 'a4', 'b1', 'b3', 'b9', 'cut_Ideal', 'color_G', 'color_H'] 

X_trn = trn.drop(columns=['outcome'])
X_trn = X_trn[selected_features]
y_trn = trn['outcome']

# XGBoost regressor parameters 
XGB_params = {'scale_pos_weight': 1, 
              'eta': 0.31,
              'gamma': 40.006000, 
              'max_depth': 2, 
              'n_estimators': 1000, 
              'reg_lambda': 100, 
              'reg_alpha': 637, 
              'subsample':0.8, 
              'colsample_bynode':0.8, 
              'colsample_bylevel': 0.9, 
              'colsample_bytree' : 0.9, 
              'random_state':42}
xgb_model = XGBRegressor(**XGB_params)

model = BaggingRegressor(estimator=xgb_model, n_estimators=16, random_state=42)

# Train Bagging Regressor 
model.fit(X_trn, y_trn)

# Test set predictions
X_tst = X_tst[selected_features]
yhat_lm = model.predict(X_tst)

# Format submission:
# This is a single-column CSV with nothing but your predictions
out = pd.DataFrame({'yhat': yhat_lm})
out.to_csv('CW1_submission_K23036967.csv', index=False) # Please use your k-number here

################################################################################

# At test time, we will use the true outcomes
y_tst = pd.read_csv('CW1_test_with_true_outcome.csv') # You do not have access to this

# This is the R^2 function
def r2_fn(yhat):
    eps = y_tst - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_tst - y_tst.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

# How does the linear model do?
print(r2_fn(yhat_lm))




