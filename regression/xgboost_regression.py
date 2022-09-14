#https://machinelearningmastery.com/xgboost-for-regression/

# check xgboost version
import xgboost
print(xgboost.__version__)

pass

# create an xgboost regression model
model = xgboost.XGBRegressor()


# create an xgboost regression model
model = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

# load and summarize the housing dataset
from pandas import read_csv
from matplotlib import pyplot
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# summarize shape
print(dataframe.shape)
# summarize first few lines
print(dataframe.head())

# split data into input and output columns
X, y = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]


...
# define model
model = xgboost.XGBRegressor()