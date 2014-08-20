from os import walk
path = "/Users/Mohsen/Documents/nlpy_mohsen/lsq/output/MATLAB/"
l = list(walk(path))
dirs = l[0][1][:-1]
for dir in dirs:
    print path+str(dir)+'/'

