import sys
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load Data
fname = str(sys.argv[1])
print("Load {}".format(fname))
X, y = pickle.load(open(fname, "rb"))
X = sparse.csr_matrix(X)

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1), test_size=0.1, random_state=42)
print("train:{}\ttest:{}".format(len(y_train), len(y_test)))
print("n_features:{}".format(X_train.shape[1]))

# Setting Parameters
tuned_params = [{"C":[0.1, 1, 10, 100], "gamma":[0.001, 0.01, 0.1, 1]}]
score = "r2"
svr = GridSearchCV(
        SVR(kernel="rbf"),
        tuned_params,
        cv=5,
        scoring=score,
        n_jobs=-1)

# Train
svr.fit(X_train, y_train)

# Report
print("Grid scores")
print(svr.grid_scores_)
print("Best params")
print(svr.best_params_)

# Evaluate
p_test = svr.predict(X_test)
test_corr = np.corrcoef(y_test, p_test)[0, 1]
test_rmse = np.sqrt(mean_squared_error(y_test, p_test))
test_r2 = r2_score(y_test, p_test)
print("RMSE:{}".format(test_rmse))
print("Corr:{}".format(test_corr))
print("R2:{}".format(test_r2))
