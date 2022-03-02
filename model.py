from typing import final
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

url = "https://raw.githubusercontent.com/jhenryshaw4/PS2HouseDataExercise/master/train.csv"
data = pd.read_csv(url)
salePrice = data['SalePrice']
#keep only numbers
numeric = data.select_dtypes(include=[np.number])
numeric_fixed = numeric.dropna(axis=1)
train = numeric_fixed.iloc[0:1000,:]

corr = train.corr()
#cols = train[0:37].index
#cols = corr['SalePrice'].sort_values(ascending=False)[0:37].index
X = train
#
Y = train['SalePrice']
X_fixed = X.drop(['SalePrice'], axis = 1)

#make model
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_fixed, Y)

# initial_test_data = numeric_fixed.iloc[1000:1200,:]
# Y2 = initial_test_data['SalePrice']
# X2 = initial_test_data
# X2_fixed = X2.drop(['SalePrice'], axis=1)
# test_predictions = model.predict(X2_fixed)
# print(len(test_predictions))

test_data = pd.read_csv("https://raw.githubusercontent.com/johntango/PS2HouseDataExercise/master/test.csv").select_dtypes(include=[np.number])
test_values = test_data.dropna(axis=1)
final_predictions = model.predict(test_values)

# with open('predictions.csv', 'w') as f:
#     writer = csv.writer()
#     writer.writerow(['Id', 'SalePrice'])
#     writer.writerows(final_predictions)
#answer = {'Id': 0, 'SalePrice': final_predictions[0]}
dict = {}
# for i in range (len(final_predictions)):
#     dict[i] = final_predictions[i]
nums = []
for i in range(1001, 1460):
    nums.append(i)
# row = pd.Series(nums)
# df = pd.DataFrame(columns=['Id','SalePrice'], data= [nums, final_predictions])
# df.to_csv('predictions.csv')
csvfile = 'predictions2.csv'
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['Id', 'SalePrice'])
    for val in zip(nums, final_predictions):
        writer.writerow(val)