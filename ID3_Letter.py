import sys
import pandas as pd
from collections import Counter
from math import log
import numpy as np
from sklearn.base import BaseEstimator as estimator, ClassifierMixin as mix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def entropy(class1=0, class2=0, class3=0, class4=0, class5=0, class6=0, class7=0, class8=0, class9=0, class10=0,
            class11=0, class12=0, class13=0, class14=0, class15=0, class16=0, class17=0, class18=0, class19=0, class20=0,
            class21=0, class22=0, class23=0, class24=0, class25=0, class26=0):
    class_list = [class1, class2, class3, class4, class5, class6, class7, class8, class9, class10,
            class11, class12, class13, class14, class15, class16, class17, class18, class19, class20,
            class21, class22, class23, class24, class25, class26]
    final_entropy = 0
    for c in class_list:
        if c != 0:
            final_entropy += -((c / sum(class_list)) * log(c / sum(class_list), 4))
    return final_entropy


# This is our main class
class id3_tree_builder(estimator, mix):

    def __init__(self, class_col="labels"):
        self.class_col = class_col

    @staticmethod
    def score(split_s, entro, total):
        # here we calculate the entropy of each branch and add them proportionally
        # to get the total entropy of the attribute
        entro_set = [entropy(*i) for i in split_s]  # entropy of each branch
        f = lambda x, y: (sum(x) / total) * y
        result = [f(i, j) for i, j in zip(split_s, entro_set)]
        return entro - sum(result)

    @staticmethod
    def split_set(header, dataset, class_col):
        # here we split the attribute into each branch and count the classes
        df = pd.DataFrame(dataset.groupby([header, class_col])[class_col].count())
        result = []
        for i in Counter(dataset[header]).keys():
            result.append(df.loc[i].values)

        return result

    @classmethod
    def node(cls, dataset, class_col):
        entro = entropy(*[i for i in Counter(dataset[class_col]).values()])
        result = {}  # this will store the total information gain of each attribute
        for i in dataset.columns:
            if i != class_col:
                split_s = cls.split_set(i, dataset, class_col)
                g_score = cls.score(split_s, entro, total=len(dataset))  # total gain of an attribute
                result[i] = g_score
        return max(result, key=result.__getitem__)

    @classmethod
    def recursion(cls, dataset, tree, class_col):
        n = cls.node(dataset, class_col)  # finding the node that sits as the root
        branchs = [i for i in Counter(dataset[n])]
        tree[n] = {}
        for j in branchs:  # we are going to iterate over the branches and create the subsequent nodes
            br_data = dataset[dataset[n] == j]  # spliting the data at each branch
            if entropy(*[i for i in Counter(br_data[class_col]).values()]) != 0:
                tree[n][j] = {}
                cls.recursion(br_data, tree[n][j], class_col)
            else:
                r = Counter(br_data[class_col])
                tree[n][j] = max(r, key=r.__getitem__)  # returning the final class attribute at the end of tree
        return

    @classmethod
    def pred_recur(cls, tupl, t):
        # if type(t) is int:
        # return "NaN"  # assigns NaN when the path is missing for a given test case
        if type(t) is not dict:
            return t
        index = {'x-box': 1, 'y-box': 2, 'width': 3, 'high': 4, 'onpix': 5, 'x-bar': 6, 'y-bar': 7, 'x2bar': 8, 'y2bar': 9, 'xybar': 10, 'x2ybr' :11, 'xy2br': 12, 'x-ege': 13, 'xegvy': 14, 'y-ege': 15, 'yegvx': 16}
        for i in t.keys():
            if i in index.keys():
                td = tupl[index[i]]
                s = t[i].get(tupl[index[i]], 0)
                r = cls.pred_recur(tupl, t[i].get(tupl[index[i]], 0))
        return r

    # main prediction function
    def predict(self, test):
        result = []
        for i in test.itertuples():
            result.append(id3_tree_builder.pred_recur(i, self.tree_))
        return pd.Series(result)  # returns the predicted classes of a test dataset in pandas Series

    def fit(self, X, y):  # this is our main method which we will call to build the decision tree
        class_col = self.class_col  # the class_col takes the column name of class attribute
        dataset = X.assign(labels=y)
        self.tree_ = {}  # we will capture all the decision criteria in a python dictionary
        id3_tree_builder.recursion(dataset, self.tree_, class_col)
        return self


if __name__ == '__main__':
    occur = 0  # counter for cross validations performed
    avg_acc = 0.0
    final_acc_arr = []
    std_dev = 0.0
    header_row = ["labels", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]  # defining the table header info
    letter_df = pd.read_csv(r'C:\Users\Soorya\Desktop\CS6735-MachineLearning\Prog Project\letter-recognition.data',
                            delimiter=",", names=header_row)  # importing the csv as a dataframe

    ObjectColumns = letter_df.select_dtypes(include=np.object).columns.tolist()
    letter_df['labels'] = [ord(item)-64 for item in letter_df['labels']]

    while (occur < 10):
        df = letter_df.sample(frac=1)
        y = df["labels"]
        X = df.drop(["labels"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        # entropy of the entire training data set (y)
        entro_set = entropy(*[i for i in Counter(y_train).values()])
        print("The total entropy of the training set is {}".format(entro_set))
        model = id3_tree_builder()  # creating a instance for the decision_tree class
        model.fit(X_train, y_train)  # calling the fit method to create the tre
        accuracy_score(y_test, model.predict(X_test))  # the accuracy score under train-test-split
        acc_arr = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print("Accuracy Scores per ", occur + 1, "Iteration is ", acc_arr)
        for i in range(0, len(acc_arr)):
            final_acc_arr.append(acc_arr[i])
        occur += 1
    avg_acc = np.sum(final_acc_arr) / len(final_acc_arr)
    std_dev = np.std(final_acc_arr)
    print("Average Accuracy:", avg_acc)
    print("Standard Deviation: ", std_dev)