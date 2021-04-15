
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
    def __init__(self, value, type):
        self.children = dict()
        # string
        self.value = value
        self.type = type

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)


class Attribute:
    def __init__(self, values, name, type):
        self.values = values
        self.name = name
        self.type = type
        # only used for the continous attributes
        self.gain = None
        self.lessExamples = None
        self.moreExamples = None

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

# Based on the decisiontree algorithm in AIMA, the logic is the same,
# I have tried to keep variables kind of similar, and excluded some commenting


def decision_tree_learning(examples, attributes, parent_examples):

    if len(examples) == 0:
        return Tree_node(plurality_value(parent_examples), "categorical")

    elif sameClassification(examples):
        return Tree_node(examples['Survived'].iloc[0], "categorical")

    elif not attributes:
        return Tree_node(plurality_value(examples), "categorical")
    else:
        attributes = findBestSplits(attributes, examples)
        A = findA(examples, attributes)

        tree = Tree_node(A, A.type)
        attributesCopy = attributes.copy()
        attributesCopy.remove(A)

        if A.type == "categorical":
            for v_k in A.values:
                exs = examples[examples[A.name] == v_k]
                subtree = decision_tree_learning(exs, attributesCopy, examples)
                tree.children[v_k] = subtree
        # if it is a continous variable
        else:  # must add a sub tree for each split, i know its only two splits
            exsLess = A.lessExamples
            subtree = decision_tree_learning(exsLess, attributesCopy, examples)
            tree.children["<"+str(A.values)] = subtree

            exsMore = A.moreExamples
            subtree2 = decision_tree_learning(
                exsMore, attributesCopy, examples)
            tree.children[">"+str(A.values)] = subtree2

        return tree

# Find the best splittings points for the continous variables
# If it is a categorical attribute, I just add it to the list without further explanation


def findBestSplits(attributes, examples):
    newAttributes = []
    for attribute in attributes:
        if attribute.type == "categorical":
            newAttributes.append(attribute)
        else:
            continousNodeWithSplit = findSplittingPoint(attribute, examples)
            newAttributes.append(continousNodeWithSplit)
    return newAttributes

# Only find splitting points between rows where classification is different
# If we come to a point where all the remaining examples have same classification
# The split will become the attributes first value, which explains the <0 and >0 splits.


def findSplittingPoint(attribute, examples):
    # sort the  examples descending, based on the attribute
    sortedExample = examples.sort_values(by=[attribute.name])
    bestAttribute = None
    bestGain = -1
    lastRow = None
    lastRowValue = None

    entropyWholeDataSet = entropyContinous(examples)
    for index, row in sortedExample.iterrows():
        # must have different classification
        # first row
        if lastRow == None:
            lastRow = row['Survived']
            lastRowValue = float(row[attribute.name])
            bestAttribute = Attribute(
                lastRowValue, attribute.name, attribute.type)
            continue
        # same classification - update values
        if row['Survived'] == lastRow:
            lastRowValue = float(row[attribute.name])
            continue
        # different classification - split and make attribute
        if row['Survived'] != lastRow:
            lastRow = row['Survived']
            splitPoint = (lastRowValue + float(row[attribute.name]))/2
            # make a new attribute object, with the splitpoint as the value
            candidateAttribute = Attribute(
                splitPoint, attribute.name, attribute.type)
            candidateGain = importanceContinous(
                examples, candidateAttribute, entropyWholeDataSet)
            candidateAttribute.gain = candidateGain
            lastRowValue = float(row[attribute.name])

            if candidateGain < 0:
                raise Exception(
                    "The gain can't be negative, you have made a major mistake")
            # if this splittingpoint causes better gain than previous:
            # We update bestattribute and bestGain
            if candidateGain > bestGain:
                bestAttribute = candidateAttribute
                bestGain = candidateGain

    return bestAttribute


def findA(examples, attributes):
    maxInfoGain = -100
    bestA = None
    # must implement another importance function
    for attribute in attributes:
        if attribute.type == "categorical":
            gain = importance(examples, attribute)
        else:
            gain = attribute.gain
        if gain > maxInfoGain:
            maxInfoGain = gain
            bestA = attribute
    if bestA == None:
        raise Exception("no best A was found, you have made a major mistake")
    return bestA


def entropyContinous(examples):
    totalSurvived = int(examples.count()[0])
    noSurvived = (examples.Survived == 0).sum()
    survived = (examples.Survived == 1).sum()
    return entropy(noSurvived/totalSurvived) + entropy(survived/totalSurvived)


def entropy(inputValue):
    if inputValue == 0 or inputValue == 1:
        return 0
    return -(inputValue)*log2(inputValue)


# Sorry for a messy code
# It just calculates the gain, based on the splittingpoint

# It updates the continous attributes variables: gain, lessExamples and moreExamples
# This is for later usage so we don't have to calculate that again
def importanceContinous(examples, attribute, entropyWholeDataSet):
    totalSurvived = int(examples.count()[0])
    informationLoss = 0
    # lessthanExample
    lessExamples = examples[examples[attribute.name] <= attribute.values]
    # Put the example in the object, so i don't have to calculate it again
    attribute.lessExamples = lessExamples
    numValueLess = lessExamples.count()[0]
    if numValueLess == 0:
        informationLoss += 0
    else:
        noSurvivedValueLess = lessExamples[lessExamples['Survived'] == 0].count()[
            0]
        survivedValueLess = lessExamples[lessExamples['Survived'] == 1].count()[
            0]
        informationLoss += (numValueLess/totalSurvived)*(entropy(
            noSurvivedValueLess/numValueLess)+entropy(survivedValueLess/numValueLess))

    # moreThanExamples
    moreExamples = examples[examples[attribute.name] > attribute.values]
    attribute.moreExamples = moreExamples
    numValueMore = moreExamples.count()[0]
    if numValueMore == 0:
        informationLoss += 0
    else:
        noSurvivedValueMore = moreExamples[moreExamples['Survived'] == 0].count()[
            0]
        survivedValueMore = moreExamples[moreExamples['Survived'] == 1].count()[
            0]
        informationLoss += (numValueMore/totalSurvived)*(entropy(
            noSurvivedValueMore/numValueMore)+entropy(survivedValueMore/numValueMore))
    return entropyWholeDataSet - informationLoss

# calculates the gain of a categorical attribute


def importance(examples, attribute):
    totalSurvived = int(examples.count()[0])
    noSurvived = (examples.Survived == 0).sum()
    survived = (examples.Survived == 1).sum()
    entropyDataSet = entropy(noSurvived/totalSurvived) + \
        entropy(survived/totalSurvived)
    informationLoss = 0
    for value in attribute.values:
        numValue = examples[examples[attribute.name] == value].count()[0]
        if numValue == 0:
            continue
        else:

            noSurvivedValue = examples[(examples[attribute.name] == value) & (
                examples.Survived == 0)].count()[0]

            survivedValue = examples[(examples[attribute.name] == value) & (
                examples.Survived == 1)].count()[0]
            informationLoss += (numValue/totalSurvived) * \
                (entropy(survivedValue/numValue)+entropy(noSurvivedValue/numValue))
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
        attributes.append(Attribute([1, 2, 3], "Pclass", "categorical"))
        attributes.append(Attribute(["male", "female"], "Sex", "categorical"))
        attributes.append(
            Attribute(['C', 'Q', 'S'], "Embarked", "categorical"))
        return attributes
    if type == "continous":
        attributes = []
        attributes.append(Attribute(None, "SibSp", "continous"))
        attributes.append(Attribute(None, "Parch", "continous"))
        attributes.append(Attribute(None, "Fare", "continous"))
        return attributes

# Just a helping function that prints the tree in dictionary format, in the terminal


def printDecisionTree(decision_tree, index):
    print("node ;", index, " is ", decision_tree.value)
    print("dictionary", decision_tree.children)
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

# helping function to find out if a row is correct or not
# I decided to split the attributes on "<=" instead of ">=". This won't have a huge impact


def correctAnswer(decision_tree, index, row):

    if (str(decision_tree.value) == "0"):
        if row['Survived'] == 0:
            return True
        else:
            return False
    if (str(decision_tree.value) == "1"):
        if row['Survived'] == 1:
            return True
        else:
            return False
    else:
        if decision_tree.type == "categorical":
            value = row[str(decision_tree.value)]
        else:
            if float(row[str(decision_tree.value)]) <= float(decision_tree.value.values):
                value = "<"+str(decision_tree.value.values)
            else:
                value = ">"+str(decision_tree.value.values)
        return correctAnswer(decision_tree.children[value], index, row)

# Helping function to plot the tree


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
    listofAttribtues = ["Survived", "Pclass",
                        "Sex", "SibSp", "Parch", "Fare", "Embarked"]
    dfTrain = dfTrain[listofAttribtues]

    dfTest = pd.read_csv("titanic/test.csv", sep=",", header=0)
    dfTest = dfTest[listofAttribtues]

    attributes = makeAttributes("categorical")
    attributes.extend(makeAttributes("continous"))

    decision_tree = decision_tree_learning(dfTrain, attributes, None)
    printDecisionTree(decision_tree, 1)

    print("correct rows/accuracy   : ", accuracy(decision_tree, dfTest))
    g = Digraph('G')
    makeTreePlot(decision_tree, 0, True)
    g.view()
