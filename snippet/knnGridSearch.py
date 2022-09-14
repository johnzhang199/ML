#Parameter grid search with xgboost
#feature engineering is not so useful and the LB is so overfitted/underfitted
#so it is good to trust your CV

#go xgboost, go mxnet, go DMLC! http://dmlc.ml

#Credit to Shize's R code and the python re-implementation
from sklearn import neighbors
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

from GridSearchFunc import gridSearch
from get_data import get_data


X, y = get_data()
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp=SimpleImputer(missing_values=np.NaN)
X2=pd.DataFrame(imp.fit_transform(X))
X2.columns=X.columns
X2.index=X.index
stat = y.to_frame().describe()

model = neighbors.KNeighborsClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have
#much fun of fighting against overfit
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {
    # 'kernel': ['linear'],#, 'rbf'],
    'algorithm': ['auto'],
}

gridSearch(model, parameters, X2, y)