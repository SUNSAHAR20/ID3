import sys
import pandas as pd
from collections import Counter
from math import log
import numpy as np
from sklearn.base import BaseEstimator as estimator, ClassifierMixin as mix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def entropy(class1=0, class2=0):
    class_list = [class1, class2]
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
        index = {'cap-shape': 1, 'cap-surface': 2, 'cap-color': 3, 'bruises': 4, 'odor': 5, 'gill-attachment': 6,
                 'gill-spacing': 7, 'gill-size': 8, 'gill-color': 9, 'stalk-shape': 10, 'stalk-root': 11, 'stalk-surface-above-ring': 12, 'stalk-surface-below-ring': 13,
    'stalk-color-above-ring': 14, 'stalk-color-below-ring': 15, 'veil-type': 16, 'veil-color': 17, 'ring-number': 18, 'ring-type': 19, 'spore-print-color': 20,
    'population': 21, 'habitat': 22}
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
    header_row = ["labels", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",  "gill-attachment",
    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color",
    "population", "habitat"]  # defining the table header info
    mushroom_df = pd.read_csv(r'C:\Users\Soorya\Desktop\CS6735-MachineLearning\Prog Project\mushroom.data',
                            delimiter=",", names=header_row)  # importing the csv as a dataframe

    mushroom_df.replace(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],inplace = True)

    mushroom_df["stalk-root"].replace(["?"], ['0'], inplace = True)
    mushroom_df["stalk-root"] = mushroom_df['stalk-root'].astype(int)
    average = mushroom_df["stalk-root"].mean()
    mushroom_df["stalk-root"].replace([0], [average], inplace = True)

    mushroom_df["labels"] = mushroom_df['labels'].astype(int)
    mushroom_df["cap-shape"] = mushroom_df['cap-shape'].astype(int)
    mushroom_df["cap-surface"] = mushroom_df['cap-surface'].astype(int)
    mushroom_df["cap-color"] = mushroom_df['cap-color'].astype(int)
    mushroom_df["bruises"] = mushroom_df['bruises'].astype(int)
    mushroom_df["odor"] = mushroom_df['odor'].astype(int)
    mushroom_df["gill-attachment"] = mushroom_df['gill-attachment'].astype(int)
    mushroom_df["gill-spacing"] = mushroom_df['gill-spacing'].astype(int)
    mushroom_df["gill-size"] = mushroom_df['gill-size'].astype(int)
    mushroom_df["gill-color"] = mushroom_df['gill-color'].astype(int)
    mushroom_df["stalk-shape"] = mushroom_df['stalk-shape'].astype(int)
    mushroom_df["stalk-surface-above-ring"] = mushroom_df['stalk-surface-above-ring'].astype(int)
    mushroom_df["stalk-surface-below-ring"] = mushroom_df['stalk-surface-below-ring'].astype(int)
    mushroom_df["stalk-color-above-ring"] = mushroom_df['stalk-color-above-ring'].astype(int)
    mushroom_df["stalk-color-below-ring"] = mushroom_df['stalk-color-below-ring'].astype(int)
    mushroom_df["veil-type"] = mushroom_df['veil-type'].astype(int)
    mushroom_df["veil-color"] = mushroom_df['veil-color'].astype(int)
    mushroom_df["ring-number"] = mushroom_df['ring-number'].astype(int)
    mushroom_df["ring-type"] = mushroom_df['ring-type'].astype(int)
    mushroom_df["spore-print-color"] = mushroom_df['spore-print-color'].astype(int)
    mushroom_df["population"] = mushroom_df['population'].astype(int)
    mushroom_df["habitat"] = mushroom_df['habitat'].astype(int)

    while (occur < 10):
        df = mushroom_df.sample(frac=1)
        y = df["labels"]
        X = df.drop(["labels"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
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