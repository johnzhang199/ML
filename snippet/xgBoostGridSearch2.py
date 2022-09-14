#Parameter grid search with xgboost
#feature engineering is not so useful and the LB is so overfitted/underfitted
#so it is good to trust your CV

#go xgboost, go mxnet, go DMLC! http://dmlc.ml

#Credit to Shize's R code and the python re-implementation
import xgboost as xgb
from GridSearchFunc import gridSearch
from get_data import get_data

X, y = get_data()

stat = y.to_frame().describe()

xgb_model = xgb.XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have
#much fun of fighting against overfit
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {
    'booster': ['gbtree'],#,'gblinear'],
    # 'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              # 'learning_rate': [0.01,0.02,0.03,0.04,0.05], #so called `eta` value
              'learning_rate': [0.19],#[0.01 * x for x in range(1,21)],#[0.01,0.02,0.03,0.04,0.050.05], #so called `eta` value
                'min_child_weight': [17],#[ x for x in range(1,21)],
              'max_depth': [4],#[ x for x in range(3,11)],#[6],
              'gamma': [0],#[ 0.01*x for x in range(30)],#[11],
              'silent': [1],
              'subsample': [0.9],#[ 0.1*x for x in range(5,11)],#[0.8],
              'colsample_bytree': [0.7],#[ 0.1*x for x in range(5,11)],#
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337],
               'scale_pos_weight' : [1/stat.loc['mean'][0]]}

gridSearch(xgb_model, parameters, X, y)