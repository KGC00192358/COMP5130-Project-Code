#!/usr/bin/env python3
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import random

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import time

import numpy as np
def trainAndTest(train, test, train_labels, test_labels, label_names, labels, feature_names, features, run_number):
    incorrect_preds = {}
    bad_preds_file = open("BadPredictionsOut.txt", "a")
    
    gnb = GaussianNB()

    model = gnb.fit(train, train_labels)
    preds = gnb.predict(test)
    for i in range(len(preds)):
        if preds[i] != test_labels[i]:
            incorrect_preds[i] = "Tumor Number " + str(i) + " was predicted to be " + str(label_names[preds[i]]).upper() + " but was actaully " + str(label_names[test_labels[i]]).upper() + "\n"
    bad_preds_file.write("{:*^20}\n".format("Run {}".format(run_number)))
    bad_preds_file.write("Number of Incorrect Preditions {}\n".format(len(incorrect_preds)))
    if len(incorrect_preds) > 0:
        for key in incorrect_preds:
            bad_preds_file.write(incorrect_preds[key])
    bad_preds_file.write("{:*^20}\n\n".format("END RUN {}".format(run_number)))
    mat = confusion_matrix(preds, test_labels)
    return accuracy_score(test_labels, preds), mat, label_names, len(incorrect_preds)

def main():
    mean_accuracy = 0
    mat = np.empty(shape=(2, 2)).tolist()
    names = []
    summary_file = open("BenchmarkOutputSummary.txt", "w")

    #load data
    data = load_breast_cancer()


    #organize data
    label_names = data['target_names']
    labels = data['target']
    feature_names = data['feature_names']
    features = data['data']
    max_fuckups = 0
    min_fuckups = 1000000000
    summary_list = []
    max_accuracy = 0
    min_accuracy = 10000000

    #print(feature_names)
    print("Number of tumors {:>7}".format(569))
    print("Number of Features {:>6}".format(len(feature_names)))
    print("Total Data Points {:>7}".format(569*len(feature_names)))
    print("Total Predictions {:>7}".format("~180000"))
    confusion_matrix = [0]
    graph_names = []
    for i in range(1, 1000):
        # Split our data
        train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=random.randint(1,1000))

        #print("{:*^20}".format("Run # {}:".format(i)))
        run_accuracy, tmp_mat, names, num_incorrect_preds = trainAndTest(train, test, train_labels, test_labels, label_names, labels, feature_names, features, i)
        mean_accuracy += run_accuracy
        if num_incorrect_preds > max_fuckups:
            max_fuckups = num_incorrect_preds
            min_accuracy = run_accuracy
        if num_incorrect_preds < min_fuckups:
            min_fuckups = num_incorrect_preds
            max_accuracy = run_accuracy
        if len(confusion_matrix) == 1:
            confusion_matrix = tmp_mat
        else:
            confusion_matrix = confusion_matrix + tmp_mat
        #print("{:*^20}".format("Run Accuracy: {}".format(run_accuracy)))
        summary_list.append("Run {0:<10} Accuracy: {1:<15}\t{2:>15} wrong predictions\n".format(i, run_accuracy, num_incorrect_preds))
    mean_accuracy = mean_accuracy/1000
    print("\nTotal Mean Accuracy for 1000 runs: {}\n".format(mean_accuracy))
    print("Highest Number of wrong predictions: {0:<10} Accuracy: {1:<10}\nLowest Number of wrong predictions: {2:<11} Accuracy: {3:<10}".format(max_fuckups, min_accuracy, min_fuckups, max_accuracy))
    summary_file.write("{:*^30}\n".format(" SUMMARY "))
    summary_file.write("Total Mean Accuracy for 1000 runs: {}\n".format(mean_accuracy))
    summary_file.write("Highest Number of wrong predictions: {0}\n Lowest Number of wrong predictions: {1}\n".format(max_fuckups, min_fuckups))
    summary_file.write("{:*^30}\n\n".format(" END SUMMARY "))
    summary_file.write("{:*^30}\n".format(" INDIVIDUAL RUN DATA "))
    for summary in summary_list:
        summary_file.write(summary)
    sns.heatmap(confusion_matrix, square=True, annot=True, fmt='d', cbar=False, xticklabels=names, yticklabels=names)

    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.show()

main()
