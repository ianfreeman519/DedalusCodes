import numpy as np
import matplotlib.pyplot as plt
import csv

# Br resolution and N2 resolutions from the original creation of the data
resolutionBr, resolutionN2 = 50, 50

masterlist = []
# Reading the file into a list
with open("eigenvaluesVsN2andBrwithlengths.txt", newline = '') as lines:
    linereader = csv.reader(lines, delimiter="\t")
    for line in linereader:
        masterlist.append(line)

# Ignoring the first row bc comments/headers... Casting into float... transposing
masterlist = np.array(masterlist[1:]).astype(float).T

indexN2, indexBr, indexvA, indexl1, indexl2, indexl3, indexl4, indexl5, indexlen = 0, 1, 2, 3, 4, 5, 6, 7, 8

N2list = masterlist[indexN2,:]
brlist = masterlist[indexBr,:]
vAlist = masterlist[indexvA,:]
l1list = masterlist[indexl1,:]
l2list = masterlist[indexl2,:]
l3list = masterlist[indexl3,:]
l4list = masterlist[indexl4,:]
l5list = masterlist[indexl5,:]
lenlist = masterlist[indexlen,:]

criticalindices = []
for i, lenkr in enumerate(lenlist):
    if i != 0 and lenlist[i-1] != lenlist[i]:
        criticalindices.append(i)

plt.scatter(N2list[criticalindices], brlist[criticalindices])
plt.ion()
plt.show()
