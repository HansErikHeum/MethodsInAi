
# importerer filen og bla bla bla
import pandas as pd
import numpy as np
from math import *
from graphviz import *
import uuid

# In this task i chose to implement tree node as an object
# This will eventually be our tree. The keys in the children-dictionary, represents the
# nodes edges, and the corresponding value represents its child node.


class Tree_node:
    def __init__(self, value):
        self.children = dict()
        self.value = value

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)


class Attribute:
    def __init__(self, values, name):
        self.values = values
        self.name = name
    # represented as a string object

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

# Based on the decisiontree algorithm in AIMA, the logic is the same,
# I have tried to keep variables kind of similar, and excluded some commenting


def decision_tree_learning(examples, attributes, parent_examples):
    if len(examples) == 0:
        return Tree_node(plurality_value(parent_examples))

    elif sameClassification(examples):
        print(examples['Survived'].iloc[0])
        return Tree_node(examples['Survived'].iloc[0])
    elif not attributes:
        return Tree_node(plurality_value(examples))
    else:

        A = findA(examples, attributes)
        tree = Tree_node(A)

        # must make a subtree for each of the currently best A's values/edges.
        for v_k in A.values:
            attributesCopy = attributes.copy()
            # remove the best a from the list of attributes
            attributesCopy.remove(A)

            # exs are all the examples that suits the current values condition
            exs = examples[examples[A.name] == v_k]
            subtree = decision_tree_learning(exs, attributesCopy, examples)
            # add a branch to tree with label A=V_K and a subtree
            tree.children[v_k] = subtree
        return tree


def findA(examples, attributes):
    maxInfoGain = -1
    bestA = None
    # find the attribute that gives the best information gain on the current examples
    for attribute in attributes:
        gain = importance(examples, attribute)
        if gain > maxInfoGain:
            maxInfoGain = gain
            bestA = attribute
    return bestA


def entropy(inputValue):
    # log2(0) is undefined, but would have been 0 anyways - so return 0
    if inputValue == 0:
        return 0
    return -(inputValue)*log2(inputValue)


def importance(examples, attribute):
    totalSurvived = int(examples.count()[0])
    noSurvived = (examples.Survived == 0).sum()
    survived = (examples.Survived == 1).sum()
    # find the entropy for the whole dataset, before splitting the examples
    entropyDataSet = entropy(noSurvived/totalSurvived) + \
        entropy(survived/totalSurvived)
    informationLoss = 0
    # find the informationLoss for each of the attribute's values
    for value in attribute.values:
        numValue = examples[examples[attribute.name] == value].count()[0]
        survivedValue = examples[(examples[attribute.name] == value) & (
            examples.Survived == 0)].count()[0]

        noSurvivedValue = examples[(examples[attribute.name] == value) & (
            examples.Survived == 1)].count()[0]
        informationLoss += (numValue/totalSurvived) * \
            (entropy(survivedValue/numValue)+entropy(noSurvivedValue/numValue))
    # return the information gain, which is this formula:
    return entropyDataSet - informationLoss


def sameClassification(examples):
    a = examples['Survived'].to_numpy()
    return (a[0] == a).all()

# return '0' og '1' based on one the one with the most remaining values.


def plurality_value(examples):
    examples = examples.Survived.value_counts().index.tolist()
    return str(examples[0])


# just a helping function to make a list of attribute-objects
def makeAttributes(type):
    if type == "categorical":
        attributes = []
        attributes.append(Attribute([1, 2, 3], "Pclass"))
        attributes.append(Attribute(["male", "female"], "Sex"))
        # attributes.append(Attribute(0, ['C', 'Q', 'S'], "Embarked"))
        return attributes
    # not really relevant, made a new class for 1b) instead.
    if type == "continous":
        pass

# Just a helping function that prints the tree in dictionary format, in the terminal


def printDecisionTree(decision_tree, index):
    print("node ;", index, " is ", decision_tree.value)
    print("dictionaryyyy", decision_tree.children)
    for key in decision_tree.children:
        printDecisionTree(decision_tree.children[key], index+1)

# Helping function to calcualte how accurate my decision-tree is


def accuracy(trained_tree, testExample):
    totalRows = testExample.count()[0]
    correctRows = 0
    # trained tree has a node and a dictionary
    for index, row in testExample.iterrows():
        if correctAnswer(trained_tree, index, row):
            correctRows += 1
    return (correctRows/totalRows)

# returns true if the decision-tree finds the correct value, false is not
# If we are not yet at an end node, we iterate further down


def correctAnswer(decision_tree, index, row):
    if (decision_tree.value == '0'):
        if row['Survived'] == 0:
            return True
        else:
            return False
    if (decision_tree.value == '1'):
        if row['Survived'] == 1:
            return True
        else:
            return False
    else:
        value = row[str(decision_tree.value)]

        return correctAnswer(decision_tree.children[value], index, row)

# Helping function to make the Graphviz-plot


def makeTreePlot(decision_tree, id, first):
    if first:
        id = str(uuid.uuid4())
        g.node(id, str(decision_tree.value))
    if (not decision_tree.children):
        return None
    for key in decision_tree.children:
        newNodeId = str(uuid.uuid4())
        g.node(newNodeId, str(decision_tree.children[key].value))
        g.edge(id, newNodeId, label=str(key))
        makeTreePlot(decision_tree.children[key], newNodeId, False)


"""The input files 'titanic/train.csv' and 'titanic/test.csv' must be changes to match where you have located the files"""
if __name__ == "__main__":
    dfTrain = pd.read_csv("titanic/train.csv", sep=",", header=0)
    dfTrain = dfTrain[["Survived", "Pclass", "Sex"]]

    dfTest = pd.read_csv("titanic/test.csv", sep=",", header=0)
    dfTest = dfTest[["Survived", "Pclass", "Sex"]]

    attributes = makeAttributes("categorical")
    decision_tree = decision_tree_learning(dfTrain, attributes, None)
    printDecisionTree(decision_tree, 1)
    print("\n Accuracy   :", accuracy(decision_tree, dfTest))
    g = Digraph('G')
    makeTreePlot(decision_tree, 0, True)
    g.view()
