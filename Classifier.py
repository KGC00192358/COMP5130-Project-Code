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

# Sorts data into my two nominal bins
# i,e Radius_Size and Texture
# Size is three bins, small medium and larg
# Texture is smooth and rough
def dataPrep(feature_name_to_all_values_dict):
    bin_dict = {}
    feature_name = 'mean radius'

    for feature_name in feature_name_to_all_values_dict:
        feature_name_to_all_values_dict[feature_name].sort()

        cur_sorted_feature_list = feature_name_to_all_values_dict
        name_string1 = "{} small".format(feature_name)
        name_string2 = "{} medium".format(feature_name)
        name_string3 = "{} large".format(feature_name)
        bin_dict[name_string1] = [cur_sorted_feature_list[feature_name][0], cur_sorted_feature_list[feature_name][188]]
        bin_dict[name_string2] = [cur_sorted_feature_list[feature_name][189], cur_sorted_feature_list[feature_name][377]]
        bin_dict[name_string3] = [cur_sorted_feature_list[feature_name][378], cur_sorted_feature_list[feature_name][568]]
    return bin_dict

def binSummary(tumors_with_bin_information, labels, label_names, bins):
    value_string = ""
    for tumor in range(len(tumors_with_bin_information)):
        value_string += "Tumor {}: \n".format(tumor)
        value_string += "\t{:<36}{:<15}\n\n".format("Label:", "Range:")
        for bin_information_dict_index in range(len(tumors_with_bin_information[tumor])):
            bin_name = tumors_with_bin_information[tumor][bin_information_dict_index][0]
            minimum = bins[bin_name][0]
            maximum = bins[bin_name][1]
            value_string += "\t{:<35} {} - {} \n".format(bin_name, minimum, maximum)
        value_string += "\nDiagnosis: {}\n\n".format(label_names[labels[tumor]].upper())

    return value_string

#This is where we start mining out the patterns/rules
#Our minimum support is 40%
def aprioriMining(nominal_dict, nominal_values):
    min_support = 0.15
    candidate_set = {}
    pop_list = []
    pattern_set = {}
    k_set = 1
    
    for value in nominal_values:
        candidate_set[value] = 0
    for key in nominal_dict:
        candidate_set[nominal_dict[key][0]] += 1
        candidate_set[nominal_dict[key][1]] += 1
    for key in candidate_set:
        if candidate_set[key]/569 < min_support:
            pop_list.append(key)
    for key in pop_list:
        candidate_set.pop(key)
    pattern_set = candidate_set
    pop_list = []
    while(k_set < 2):
        k_set += 1
        candidate_set = {}
        for key1 in nominal_values:
            for key2 in nominal_values:
                candidate_key = key1 + ' + ' + key2
                candidate_set[candidate_key] = 0
        for key in nominal_dict:
            candidate_key = nominal_dict[key][0] + ' + ' + nominal_dict[key][1]
            candidate_set[candidate_key] += 1
        for key in candidate_set:
            if candidate_set[key]/569 < min_support:
                pop_list.append(key)
        for key in pop_list:
            candidate_set.pop(key)

        pattern_set = candidate_set
        rule_set = []
        for key in pattern_set:
            rule_set.append([key.split(" ")[0], key.split(" ")[2]])


    return rule_set
def aprioriMiningTwo(transaction_list, all_possible_transactions):
    candidate_set = {}
    pattern_set = {}
    rule_set = []
    min_support = 0.10
    k_set = 0
    for transaction in all_possible_transactions:
        candidate_set[transaction] = 0
    
    while(len(candidate_set) > 0):
        k_set += 1
        candidate_set = getPatternCount(candidate_set, transaction_list)
        candidate_set = pruning(candidate_set, min_support)
        if(len(candidate_set) == 0):
            break
        pattern_set = candidate_set
        candidate_set = generateNextPatternSet(candidate_set, k_set)
    
   
    for key in pattern_set:
        rule = ''
        for antecendent in key.split(" "):
            rule += antecendent + ' '
        rule_set.append(rule)
    return rule_set

def getPatternCount(candidate_set, transaction_list):
    for transaction_index in range(len(transaction_list)):
        for transaction in candidate_set:
            if transaction in transaction_list[transaction_index]:
                candidate_set[transaction] += 1
    return candidate_set

def pruning(candidate_set, min_support):
    tmp = candidate_set
    prune_list = []
    for key in tmp:
        if tmp[key]/569 < min_support:
            prune_list.append(key)
    for i in range(len(prune_list)):
        tmp.pop(prune_list[i])
    return tmp
def generateNextPatternSet(candidate_set, k_set):
    pattern_list = []
    new_pattern_list = []
    new_candidate_set = {}
    for key in candidate_set:
        pattern_list.append(key)

    if (k_set > 2):
        first_n_to_match = k_set - 2
        for i in range(len(pattern_list)):
            for j in range(len(pattern_list)):
                pattern_to_add = pattern_list[i] + ", " + pattern_list[j] 
                add_pattern = True
                string1 = pattern_list[i]
                string2 = pattern_list[j]
                for k in range(len(string1.split(", "))):
                    if string1.split(", ")[k] != string2.split(", ")[k]:
                        add_pattern = False
                if add_pattern:
                    new_pattern_list.append(pattern_to_add)
    else:
        for i in range(len(pattern_list)):
            for j in range(i+1, len(pattern_list)):
                if not i == j:
                    new_pattern_list.append(pattern_list[i] + ", " + pattern_list[j])

    for i in range(len(new_pattern_list)):
            new_candidate_set[new_pattern_list[i]] = 0
    return new_candidate_set





def bayesianRulesGeneration(nominal_dict, rule_set, diagnosis):
    if_then_rule_list = []

    number_of_malignant = 0 #malignat = 0
    for rule in rule_set:
        for i in range(len(nominal_dict)):
            if nominal_dict[i][0] == rule[0] and nominal_dict[i][1] == rule[1] and diagnosis[i] == 0:
                number_of_malignant += 1
        print("Rule: {} AND {}\tMalNum: {} ".format(rule[0], rule[1], number_of_malignant))
        if_then_rule_list.append("If {} and {} then {:.2f}% Benign".format(rule[0], rule[1], (1 - (number_of_malignant/569))*100))
    for rule in if_then_rule_list: 
        print(rule)


#Need Probability Benign | Rule
#Must find Prob(Rule | Benign), Prob(Benign) Prob(rule)
def bayesianRulesGenerationTwo(transaction_list, rule_set, diagnosis):
    final_rule_set = {}
    rule_to_prob_given_benign = {}
    rule_to_prob_of_rule = {}
    minimum_chance = .55 # want slightly better than 50% chance of a tumor being benign if this rule is true
    for rule in rule_set:
        rule_to_prob_given_benign[rule] = 0
        rule_to_prob_of_rule[rule] = 0
    number_of_benign = 0
    prob_benign = 0
    total = len(diagnosis)
    for i in range(len(diagnosis)):
        number_of_benign += diagnosis[i] 

    prob_benign = number_of_benign/total
    for i in range(len(diagnosis)):
        for rule in rule_to_prob_given_benign:
            if diagnosis[i] == 1 and rule.strip(', ') in transaction_list[i].strip(', '):
                rule_to_prob_given_benign[rule] += 1
        rule_to_prob_given_benign[rule] = rule_to_prob_given_benign[rule] / number_of_benign
    for rule in rule_to_prob_of_rule:
        for i in range(len(transaction_list)):
            if rule.strip(', ') in transaction_list[i].strip(', '):
                rule_to_prob_of_rule[rule] += 1
    rule_to_prob_of_rule[rule] = rule_to_prob_of_rule[rule] / len(transaction_list)

    for rule in rule_set:
        chance_benign_given_rule = ((rule_to_prob_given_benign[rule] * prob_benign)/rule_to_prob_of_rule[rule])
        if chance_benign_given_rule >= minimum_chance:
            final_rule_set[rule] = chance_benign_given_rule

    return final_rule_set





    
def verboseSummary(features, feature_names, labels, label_names, nominal_dict):
    tumor_number_to_value_string = {}
    for i in range(len(features)):
        value_string = "Tumor {0}\n".format(i)
        for j in range(len(feature_names)):
            value_string += "\t{:<24}{:>7.2f}\n".format(str(feature_names[j])+":", features[i][j])
        value_string += "\t{:<24}{:>7}\n".format("Size:", nominal_dict[i][0])
        value_string += "\t{:<24}{:>7}\n".format("Texture:", nominal_dict[i][1])
        value_string += "{:<24}{:>7}\n\n".format("Diagnosis:", label_names[labels[i]].upper())
        tumor_number_to_value_string[i] = value_string
    return tumor_number_to_value_string

def succintSummary(nominal_dict):
    output_dict = {}
    for i in range(len(nominal_dict)):
        value_string = "Tumor {} is {} with a {} texture\nsublist python".format(i, nominal_dict[i][0], nominal_dict[i][1])
        output_dict[i] = value_string
    return output_dict

def makeBinnedVersion(bin_dict, tumors_with_resonable_feature_names, feature_names, features):
    tumors_with_bin_names = {}
    for i in range(len(features)):
        for j in range(len(feature_names)):
            if i in tumors_with_bin_names:
                tumors_with_bin_names[i].append([feature_names[j], features[i][j]])
            else:
                tumors_with_bin_names[i] = [[feature_names[j], features[i][j]]]
                
    for tumor in tumors_with_bin_names:
        for feature_name_and_value_index in range(len(tumors_with_bin_names[tumor])):
            feature_name = tumors_with_bin_names[tumor][feature_name_and_value_index][0]
            feature_value = tumors_with_bin_names[tumor][feature_name_and_value_index][1]
            #print("{}: {}".format(feature_name, feature_value))
            if feature_value >= bin_dict[feature_name + " small"][0] and feature_value <= bin_dict[feature_name + " small"][1]:
                tumors_with_bin_names[tumor][feature_name_and_value_index][0] = feature_name + " small"
            elif feature_value >= bin_dict[feature_name + " medium"][0] and feature_value <= bin_dict[feature_name + " medium"][1]:
                tumors_with_bin_names[tumor][feature_name_and_value_index][0] = feature_name + " medium"
            elif feature_value >= bin_dict[feature_name + " large"][0] and feature_value <= bin_dict[feature_name + " large"][1]:
                tumors_with_bin_names[tumor][feature_name_and_value_index][0] = feature_name + " large"



    return tumors_with_bin_names

def makePredictionsBasedOnRules(rule_set, transaction_list):
    preds = [0] * len(transaction_list)
    for i in range(len(transaction_list)):
        for rule in rule_set:
            if rule.strip(', ') in transaction_list[i].strip(', '):
                preds[i] = 1
    return preds

def ruleSummaryFile(rule_set):
    rule_summary_file = open("RuleSummary.txt", "w")
    string = ''
    for rule in rule_set:
        string = 'IF '
        for ant in rule.split(", "):
            string += ("{} AND ".format(ant))
        string +=("\n\tTHEN Benign\n")
        rule_summary_file.write(string)
        
        

    

def main():
    outfile = open("ClassificationSummary.txt", "w")
    data = load_breast_cancer()
    label_names = data['target_names']
    labels = data['target']
    feature_names = data['feature_names']
    features = data['data']
    feature_name_to_all_values_dict = {}
    tumors_with_resonable_feature_names = {}
    possible_bins = []
    transaction_list = ['']*569
    classification_rules = {}
    prediction_list = []
    for i in range(len(features)):
        for j in range(len(feature_names)):
            if i in tumors_with_resonable_feature_names:
                tumors_with_resonable_feature_names[i].append([feature_names[j], features[i][j]])
            else:
                tumors_with_resonable_feature_names[i] = [[feature_names[j], features[i][j]]]
    string = "" 
    for thing in tumors_with_resonable_feature_names:
        string += "{:<20}{:<20}{:<20}\n".format("Name", "Given", "Actual")
        string += str(thing) + "\n"
        for value in tumors_with_resonable_feature_names[thing]:
            string += "\t{:<20}{:<20}{:<20}\n".format(value[0], value[1], features[thing][0])
        #print(string)
    
    for feature_name in feature_names:
        feature_name_to_all_values_dict[feature_name] = []
    for i in range(len(features)):
        for j in range(len(feature_names)):
            feature_name_to_all_values_dict[feature_names[j]].append(features[i][j])
    
    data_bins_ranges_and_labels = dataPrep(feature_name_to_all_values_dict)
    nominal_dict = makeBinnedVersion(data_bins_ranges_and_labels, tumors_with_resonable_feature_names, feature_names, features)
    
    
    
    for bin_name in data_bins_ranges_and_labels:
        possible_bins.append(bin_name)
    for i in range(len(nominal_dict)):
        for j in range(len(nominal_dict[i])):
            transaction_list[i] += nominal_dict[i][j][0] + ", "

    rule_set = aprioriMiningTwo(transaction_list, possible_bins)
    classification_rules = bayesianRulesGenerationTwo(transaction_list, rule_set, labels)
    prediction_list = makePredictionsBasedOnRules(classification_rules, transaction_list)
    ruleSummaryFile(rule_set)
    accuracy = accuracy_score(prediction_list, labels)
    print("Accuracy: {}".format(accuracy))
    write_string = "Accuracy: {}\nNumber of Rules(check RulesSummary for More Information): {}".format(accuracy, len(classification_rules))
    outfile.write(write_string)
    confusion_mat = confusion_matrix(prediction_list, labels) 
    sns.heatmap(confusion_mat, square=True, annot=True, fmt='d', cbar=False, xticklabels=(label_names), yticklabels=(label_names))
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.savefig("ClassifyerHeatMap.png")
    plt.show()

main()
