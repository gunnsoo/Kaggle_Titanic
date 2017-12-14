import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestClassifier


#訓練データ
df = pd.read_csv("train.csv").replace("male",0).replace("female",1)
df["Age"].fillna(df.Age.median(), inplace=True)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df2 = df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

#モデル作成
train_data = df2.values
xs = train_data[:, 2:]
y = train_data[:, 1]
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(xs, y)

#テストデータ
test_df = pd.read_csv("test.csv").replace("male", 0).replace("female", 1)
test_df["Age"].fillna(df.Age.median(), inplace=True)
test_df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
test_df2 = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

#予測
test_data = test_df2.values
xs_test = test_data[:, 1:]
output = forest.predict(xs_test)
print(len(test_data[:,0]), len(output))
zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)

#csv作成
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])
