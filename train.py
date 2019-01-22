# encoding: utf-8
# kaggle問題、 train/ test.csv の使用して検証する。
# 2019/01/22 17:34 : train, dropna() 追加
# train : % 
# test  : %

# データ加工・処理・分析モジュール
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl
# 機械学習モジュール
import sklearn
from sklearn import linear_model
#from sklearn.model_selection import train_test_split
import pickle
import time 

#
def conv_dammy_mat(df):
    head_mat = "matchType"
    for item in df.columns:
        df=  df.rename(columns={ item : head_mat +"_"+ item} )
    return df
#
def conv_xdata(df):
    df =df.drop("Id", axis=1)
    df =df.drop("groupId", axis=1)
    df =df.drop("matchId", axis=1)
    df =df.drop("matchType", axis=1)
    return df

# 学習データ
global_start_time = time.time()

train = pd.read_csv("train-tmp.csv") # Reading data
test  = pd.read_csv("test-tmp.csv") # Reading data
# 
train = train[: 10000]
test  = test[: 10000]
#train_sub = train[: 10000]
#train_sub.to_csv('train-tmp.csv', index=False)

train.dropna(inplace=True)
#print( train.isnull().sum() )
print( train.isnull().any() )
train.info()
#quit()

# conv, dummy val
train_mat = pd.get_dummies(train["matchType"]) 
#train= train[: 10000]
train_mat =conv_dammy_mat(train_mat )
print(train.shape )
print(train_mat.shape )

#
test_mat = pd.get_dummies(test["matchType"]) 
#train= train[: 10000]
test_mat =conv_dammy_mat(test_mat )
print(test.shape )
print(test_mat.shape )

#quit()
#print( len(train_mat.columns ))
#print( len(test_mat.columns ))
#quit()
#merge
train_mat["Id"] = train["Id"]
train_all= pd.merge( train , train_mat )
test_mat["Id"] = test["Id"]
test_all= pd.merge( test , test_mat )
#quit()
#
y_train =train_all['winPlacePerc']
y_data = y_train
#train_all.info()
x_train =train_all
x_train =conv_xdata(x_train )
x_train =x_train.drop("winPlacePerc", axis=1)

x_test = test_all
x_test = conv_xdata(test_all )
#
print(x_train.shape ,y_train.shape )
print(x_test.shape  )

#x_train.info()
#quit()
# model
# モデルのインスタンス
model = linear_model.LinearRegression()
# fit
clf = model.fit( x_train ,y_train)
print("train:",clf.__class__.__name__ ,clf.score(x_train,y_train))
#quit()

#
#pred
#x_train_sub=x_train[: 1000]
x_test_sub = x_test.drop("winPlacePerc", axis=1)
# x_train =x_train.drop("winPlacePerc", axis=1)
pred = model.predict(x_test_sub )
#pred = model.predict(x_train_sub )
print( pred[: 10] )

#csv
df = pd.DataFrame(pred, test["Id"], columns=["winPlacePerc"])
#
df.to_csv("out.csv", index_label=["Id"])
quit()

#
# 回帰、評価
# MAE
from sklearn.metrics import mean_absolute_error
a= mean_absolute_error(y_test, pred )
print("MAE =" + str(a)  )

#MSE
from sklearn.metrics import mean_squared_error
a=mean_squared_error(y_test, pred )
print("MSE =" + str(a)  )

#RMSE
from sklearn.metrics import mean_squared_error
a= np.sqrt(mean_squared_error(y_test, pred ) )
#print("RMSE=" + str(a)  )
print("RMSE=", a  )
quit()
#
#plt
#a1=np.arange(len(x_train) )
a1=np.arange(len(x_train_sub ) )
plt.plot(a1 , y_train[: 1000]  , label = "y_train")
plt.plot(a1 , pred , label = "predict")
plt.legend()
plt.grid(True)
plt.title("PUBG")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
quit()


