#!/usr/bin/env python3
import numpy.matlib
import numpy as np

data_file = open("CA-GrQc.txt", "r")
output_file = open("data_out.t", "w")
data_lines = data_file.readlines()
data_file.close()

edge_count = 0
expected_edge_count = int(data_lines[2].split(" ")[4].strip())
number_of_nodes = int(data_lines[2].split(" ")[2])
author_dict = {}
number_of_authors = 0
edge_list_author_one = []
edge_list_author_two = []

for i in range(4, len(data_lines)):
    author_id_one = data_lines[i].split("\t")[0].strip()
    author_id_two = data_lines[i].split("\t")[1].strip()
    author_dict[author_id_one] = number_of_authors
    number_of_authors = len(author_dict)
    author_dict[author_id_two] = number_of_authors
    number_of_authors = len(author_dict)
    edge_list_author_one.append(author_dict[author_id_one])
    edge_list_author_two.append(author_dict[author_id_two])

adjmatrix =[[0 for j in range(number_of_nodes + 1)] for k in range(number_of_nodes + 1)]

for i in range(len(edge_list_author_one)):
    u = edge_list_author_one[i]
    v = edge_list_author_two[i]
    print("u: " + str(u) + "\t v: " + str(v))
    print("rows: " + str(len(adjmatrix)))
    print("cols: " + str(len(adjmatrix)))
    adjmatrix[u][v] = 1

output_file.write("Matrix: \n")

for i in range(len(adjmatrix)):
    for k in range(len(adjmatrix[0])):
        print(adjmatrix[i][k], " ", end='')
    print('')
