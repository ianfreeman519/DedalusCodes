import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import csv
from mpi4py import MPI


"""
krlist = [[1, 2, 3],[1, 2, 3],[1,2],[1,2],[1,2],[1,2],[1],[1],[1],[1],[1],[1],[1]]
xlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

krlist = np.array(krlist, dtype=object)

lens = np.array([len(i) for i in krlist])
mask = np.arange(lens.max()) < lens[:, None]
newkrlist = np.empty(mask.shape, dtype=krlist.dtype)
newkrlist[mask] = np.concatenate(krlist)
krlist = newkrlist

print(krlist)
print(np.transpose(krlist))

def fillTranspose(data):
    # Takes a 2d array called data, and fills shorter entries with None, then transposes the result
    lens = np.array([len(i) for i in data])
    mask = np.arange(lens.max()) < lens[:, None]
    newdata = np.empty(mask.shape, dtype=data.type)
    newdata[mask] = np.concatenate(data)
    return np.transpose(newdata)


for y in np.real(krlist):
    print(len(xlist), len(y))
    plt.scatter(xlist[0:len(y)], y, color="black", label="ell")


krlist = np.transpose(krlist)


print(len(xlist), len(krlist[0]))
plt.scatter(xlist, krlist[0])
plt.scatter(xlist[0:len(krlist[1])], krlist[1])
plt.scatter(xlist[0:len(krlist[2])], krlist[2])


for l, krL in enumerate(krlist):
    plt.scatter(xlist[0:len(krL)], krL, label=f'l={l}')

plt.legend()
plt.show()"""

"""
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
lenlist= masterlist[indexlen,:]

masterdata = {"N2s": N2list, "Brs": brlist, "vAs": vAlist, "krs": [l1list, l2list, l3list, l4list, l5list]}
outputfile = open("savetest.p", "wb")
pickle.dump(masterdata, outputfile)
outputfile.close()

inputfile = open("savetest.p", "rb")
masterdata2 = pickle.load(inputfile)
inputfile.close()

print(masterdata2["krs"][1].shape)
"""











# masterData = np.load("simulatedKrPickled.npy")

# print(masterData.shape)
"""
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

y = np.linspace(1, 20, num=20)
x = np.linspace(1, 20, num=20)
for i, xx in enumerate(x[rank::size]):
    print(rank, i, xx)

for i, xx in enumerate(x)[1,rank::size]:
    y[i]=np.sin(xx)

print(rank, y)
"""
inputfile = open("simulatedKrPickled.pkl", "rb")
masterdata = pickle.load(inputfile)
print(masterdata["krs"])