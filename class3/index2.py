import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression as LR  # 線形回帰のモデル
# %matplotlib inline

data = pd.read_csv("class3/data/lunch_box.csv",
                   delimiter=",",
                   # index_col='datetime',
                   parse_dates=True
                   )
test = pd.read_csv('class3/data/test.csv')  # 検証データの読み込み
sample = pd.read_csv('class3/data/sample.csv', header=None)  # 提出用サンプルデータの読み込み

# データ確認
# print(data.head())

# 基本統計量
# print(data.describe())

# 検証用データ
_data = pd.get_dummies(data[['week', 'temperature']])
# print(_data.head())

y = data['y']
# print(y.head())

# モデルの定義
model = LR()

"""
# 重回帰モデルの作成
# model.fit(_data, y)
# 傾き
# model.coef_
# 切片
# model.intercept_

# 予測
# pred = model.predict(_data)
# 予測結果
# print(pred)

# https: // signate.jp/competitions/24 でモデル評価ができる
# sample[1] = pred
# sample.to_csv('class3/result/pred1.csv', index=None, header=None)
"""

"""
pred1.csvでは単回帰分析よりも精度が落ちている
以下で精度を上げる
"""

# class3/result/plod.csv
# data['y'].plot(figsize=(14, 4))

# datetimeを分割して設定
data['year'] = data['datetime'].apply(lambda x: x.split('/')[0])
data['month'] = data['datetime'].apply(lambda x: x.split('/')[1])
test['year'] = test['datetime'].apply(lambda x: x.split('-')[0])
test['month'] = test['datetime'].apply(lambda x: x.split('-')[1])

# 新規で定義したカラムのデータ型はobject型になってしまうため、int型へ変換する
data['yaer'] = data['year'].astype(np.int)
data['month'] = data['month'].astype(np.int)
test['yaer'] = test['year'].astype(np.int)
test['month'] = test['month'].astype(np.int)

"""
xData = data[['year', 'month']]
xTest = test[['year', 'month']]
yData = data['y']

model.fit(xData, yData)
model.coef_  # 傾き
model.intercept_  # 切片
pred = model.predict(xTest)
sample[1] = pred
sample.to_csv('class3/result/pred2.csv', index=None, header=None)
newPred = model.predict(xData)
data['pred'] = newPred

# trainのyとpredを引き算した結果をtrainの新たなカラムresとして代入
data['res'] = data['y'] - data['pred']

# ソートして中身を確認
data.sort_values(by='res')
"""

# 関数


def isFun(x):
    if x == 'お楽しみメニュー':
        return 1
    else:
        return 0


data['fun'] = data['remarks'].apply(lambda x: isFun(x))
test['fun'] = test['remarks'].apply(lambda x: isFun(x))
# data.sort_values(by='fun', ascending=False)[:10]
# data.sort_values(by='fun', ascending=True)[:10]

x_train = data[['yaer', 'month', 'fun', 'temperature']]
y_train = data['y']
x_test = test[['year', 'month', 'fun', 'temperature']]

newModel = LR()

newModel.fit(x_train, y_train)

newPred = newModel.predict(x_test)
# sample[1] = newPred
# sample.to_csv('class3/result/pred3.csv', index=None, header=None)


plt.show()
