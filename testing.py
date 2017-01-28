import pyraft as pr
import numpy as np
import math
# for i in range(0,10):

# 	print(i)

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


a = list(split(range(10), 3))
b = len(a[2])

print(a)
print(b)

col_M = list(split(range(10),3))
print(len(col_M[0][:]))
