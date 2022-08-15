import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

data_df6 = pd.read_csv('vars13.csv',index_col=0)
data_df6[:246]
data_df6=data_df6 [:-1]

def RandomForests(array):
    # gather data

    dataset = data_df6
    print(dataset)
    print(dataset.head())
    X = dataset.iloc[:, 0:24].values  # change depedning on number of variables
    y = dataset.iloc[:, 24].values  # change depedning on number of variables

    # train

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    from sklearn.ensemble import RandomForestRegressor

    regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    # df.to_csv ("testingdataFED4.csv")

    # from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred))

    # Feature importance

    #from matplotlib import pyplot

    #importance = regressor.feature_importances_

    #for i, v in enumerate(importance):
        #print('Feature: %0d, Score: %.5f' % (i, v))

    #pyplot.bar([x for x in range(len(importance))], importance)
    #pyplot.show()

    # prediction
    pred = [
        [array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7], array[8], array[9], array[10],
         array[11], array[12], array[13], array[14], array[15], array[16], array[17], array[18], array[19], array[20],
         array[21], array[22], array[23]]]
    finalpred = regressor.predict(pred)
    # thepred = regressor.predict_proba(pred)
    print(finalpred)
    # print(thepred)

    print("Training Accuracy = ", regressor.score(X_train, y_train))
    print("Test Accuracy = ", regressor.score(X_test, y_test))

    return finalpred[0]


arr = []
array = []

from csv import reader

with open('Book13.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        print(row)
        arr.append(row[1])
        arr.append(row[2])
        arr.append(row[3])
        arr.append(row[4])
        arr.append(row[5])
        arr.append(row[6])
        arr.append(row[7])
        arr.append(row[8])
        arr.append(row[9])
        arr.append(row[10])
        arr.append(row[11])
        arr.append(row[12])
        arr.append(row[13])
        arr.append(row[14])
        arr.append(row[15])
        arr.append(row[16])
        arr.append(row[17])
        arr.append(row[18])
        arr.append(row[19])
        arr.append(row[20])
        arr.append(row[21])
        arr.append(row[22])
        arr.append(row[23])
        arr.append(row[24])
        print(arr)
        finalpred = RandomForests(arr)
        print("HERE IS THE NUM\n\n")

        array.append(finalpred)
        print(array)

df = pd.DataFrame(array)
df.to_csv("output15.csv")
