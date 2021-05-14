# 評価基準：f1_score
# https: // scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

# pandasのimport
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# 前処理
df = pd.read_csv('data/train.csv')

# 欠損値処理
df = df.dropna()

# 重複部分の削除
df.drop_duplicates(inplace=True)

# 重複部分の削除の後に新しいインデックスを取得
df.reset_index(drop=True, inplace=True)

# 前処理
df["term"] = df["term"].str.replace('years', '')
df["employment_length"] = df["employment_length"].str.replace(
    'years', '').str.replace(
    'year', '')
df["loan_status"] = df["loan_status"].str.replace(
    'FullyPaid', '0').str.replace('ChargedOff', '1')

df["term"] = df["term"].astype(int)
df["employment_length"] = df["employment_length"].astype(int)
df["loan_status"] = df["loan_status"].astype(int)

# データフレームの分離
col_categoric = ["grade", "purpose", "application_type", "loan_status"]
df_numeric = df.drop(col_categoric, axis=1)
df_categoric = df[col_categoric]

# df_categoric内の"disease"列と、df_numericの列を横結合する
df_tmp = pd.concat([df_categoric["loan_status"], df_numeric], axis=1)

# ダミー変数化
df = pd.get_dummies(df, columns=["grade", "purpose", "application_type"])

# 基本統計量
# print(df.describe(include='all'))
"""
                 id      loan_amnt           term  ...  application_type_Joint App  loan_status_ChargedOff  loan_status_FullyPaid
count  2.289710e+05  228971.000000  228971.000000  ...               228971.000000           228971.000000          228971.000000
mean   5.570189e+07    1433.415476       3.448887  ...                    0.022627                0.194544               0.805456
std    4.789707e+07     875.149218       0.834432  ...                    0.148713                0.395850               0.395850
min    5.571600e+04     100.000000       3.000000  ...                    0.000000                0.000000               0.000000
25%    3.345535e+06     780.000000       3.000000  ...                    0.000000                0.000000               1.000000
50%    8.552937e+07    1200.000000       3.000000  ...                    0.000000                0.000000               1.000000
75%    9.230207e+07    2000.000000       3.000000  ...                    0.000000                0.000000               1.000000
max    1.264193e+08    4000.000000       5.000000  ...                    1.000000                1.000000               1.000000
"""

# データ数を取得
counts_loan_status = df["loan_status"].value_counts()

# 棒グラフによる可視化ー質的データ
# counts_loan_status.plot(kind='bar')
"""
graph/loan-status-bar.png
0(FullyPaid)>1(ChargedOff)
"""

# 数量変数のヒストグラムを表示(※figsizeオプションはグラフのサイズを指定）
# df_numeric.hist(figsize=(8, 6))
# グラフのラベルが重ならないようにレイアウトを自動調整
# plt.tight_layout()
"""
graph/numeric-hist.png
全体的に左寄り
"""

"""
仮説
債務履行者（loan_status=0）に比べ、債務不履行者（loan_status=1)のデータでは、正常な検査値の範囲（基準値）から外れるケースが多くなるため、
ヒストグラムを描いた時のピーク値（最頻値）などが、債務履行者と異なるのではないか
"""

print(df_tmp)

# グラフの表示
plt.figure(figsize=(12, 12))
for ncol, colname in enumerate(df_numeric.columns):
    plt.subplot(3, 3, ncol+1)
    sns.distplot(df_tmp.query("loan_status==0")[colname])
    sns.distplot(df_tmp.query("loan_status==1")[colname])
    plt.legend(labels=["FullyPaid", "ChargedOff"], loc='upper right')
plt.show()


plt.show()
