#Parameter grid search with xgboost
#feature engineering is not so useful and the LB is so overfitted/underfitted
#so it is good to trust your CV

#go xgboost, go mxnet, go DMLC! http://dmlc.ml

#Credit to Shize's R code and the python re-implementation

import pandas as pd
import numpy as np
import xgboost as xgb
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split

from sklearn.model_selection import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

# from sklearn.metrics import precision_recall_curve

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def gridSearch(model, parameters, X, y, test_size=0.3):





    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)
    else:
        X_train = X
        y_train = y

    # xgb_model = xgb.XGBClassifier()
    #
    # #brute force scan for all parameters, here are the tricks
    # #usually max_depth is 6,7,8
    # #learning rate is around 0.05, but small changes may make big diff
    # #tuning min_child_weight subsample colsample_bytree can have
    # #much fun of fighting against overfit
    # #n_estimators is how many round of boosting
    # #finally, ensemble xgboost with multiple seeds may reduce variance
    # parameters = {
    #     'booster': ['gbtree'],#,'gblinear'],
    #     # 'nthread':[4], #when use hyperthread, xgboost may become slower
    #               'objective':['binary:logistic'],
    #               # 'learning_rate': [0.01,0.02,0.03,0.04,0.05], #so called `eta` value
    #               'learning_rate': [0.19],#[0.01 * x for x in range(1,21)],#[0.01,0.02,0.03,0.04,0.050.05], #so called `eta` value
    #                 'min_child_weight': [17],#[ x for x in range(1,21)],
    #               'max_depth': [4],#[ x for x in range(3,11)],#[6],
    #               'gamma': [0],#[ 0.01*x for x in range(30)],#[11],
    #               'silent': [1],
    #               'subsample': [0.9],#[ 0.1*x for x in range(5,11)],#[0.8],
    #               'colsample_bytree': [0.7],#[ 0.1*x for x in range(5,11)],#
    #               'n_estimators': [5], #number of trees, change it to 1000 for better results
    #               'missing':[-999],
    #               'seed': [1337],
    #                'scale_pos_weight' : [1/stat.loc['mean'][0]]}


    clf = GridSearchCV(model, parameters, n_jobs=5,
                       cv=
                       # StratifiedKFold(
                       RepeatedStratifiedKFold(    # y_train,
                                          n_splits=5,
                           n_repeats=5
                                          # shuffle=True
                       ),
                       scoring='roc_auc',
                       verbose=2, refit=True)

    clf.fit(X_train, y_train)


    print(pd.DataFrame(clf.cv_results_))

    print(clf.best_params_)

    if test_size> 0:
        print(clf.score(X_test,y_test))
        if hasattr(clf, 'predict_proba'):
            test_probs = clf.predict_proba(X_test)[:,1]

            test_class = np.array(clf.predict(X_test))

            np.unique(test_class)

            pd.DataFrame(test_class).describe()

            #auc curve
            fpr, tpr, threshold = metrics.roc_curve(y_test, test_probs)
            roc_auc = metrics.auc(fpr, tpr)

            plt.title('Receiver Operating Characteristic:' + str(type(model)))
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            #
            # plt.figure()
            # import seaborn as sns
            # fig = sns.kdeplot(churn_testing[churn_testing['CHURN_PROBABILITY']==0]['predicted'].rename('not_churned'), shade=True, color="r")
            # fig = sns.kdeplot(churn_testing[churn_testing['CHURN_PROBABILITY']==1]['predicted'].rename('churned'), shade=True, color="g")
            # print(1)
            # time.sleep(1)
            # fig.legend(['not_churned', 'churned'])
            # print(2)
            # time.sleep(1)
            # plt.title('Predicted Probability Distribution using only '+VAR_X)
            # print(3)
            # time.sleep(1)
            # # plt.show()
            # # print(4)
            # # plt.figure()
            # #
            # # churn_testing[churn_testing['STATUS_VAL']==0][['predicted']].rename(columns={'predicted': 'Rejected'}).plot.kde(legend =True)
            # # churn_testing[churn_testing['STATUS_VAL']==1][['predicted']].rename(columns={'predicted': 'Approved'}).plot.kde(legend =True)
            # #
            # # plt.show()
            # #
            # time.sleep(1)
            #
            # plt.figure()

            # disp = PrecisionRecallDisplay( \
            #     churn_testing['STATUS_VAL'], churn_testing['predicted'])
            # disp.plot()


            PrecisionRecallDisplay.from_predictions( \
               y_test, test_probs)
            plt.show()
            #             plt.title('Precision/Recall using only tenure in days')

            #let's do the probability curve of tenure
            #   curve = pd.DataFrame.from_dict({VAR_X: range(0,10000)})
            # curve['predicted churn probability'] = result.predict(curve)
            # curve.set_index(VAR_X, inplace=True)
            # curve.plot()

            # train_class = np.array(clf.predict(X_train))
            # np.unique(train_class)
            #
            # #trust your CV!
            # best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
            # print('Raw AUC score:', score)
            # for param_name in sorted(best_parameters.keys()):
            #     print("%s: %r" % (param_name, best_parameters[param_name]))
            #
            #
            #
            # sample = pd.read_csv('../input/sample_submission.csv')
            # sample.QuoteConversion_Flag = test_probs
            # sample.to_csv("xgboost_best_parameter_submission.csv", index=False)
    return clf