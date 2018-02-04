#/usr/bin/python

import numpy as np
import modules.pyplot_ratio_bars as plt



def softmax(abc):
    abc_exp = np.exp(abc)
    return abc_exp / np.sum(abc_exp)


def compare(ary):
    data = np.array(ary)
    plt.plot(data, softmax(data), True)




#data
compare([0, 1, 0])
compare([1, 2, 1])
compare([2, 4, 2])
compare([3, 6, 3])

plt.show()
