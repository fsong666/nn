import numpy as np
import matplotlib.pyplot as plt

# x = np.arange(0,  3  * np.pi,  0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)
# plt.subplot(2,  1,  1)
# plt.plot(x, y_sin)
# plt.title('Sine')
#
# plt.subplot(2,  1,  2)
# plt.plot(x, y_cos)
# plt.title('Cosine')
# # 展示图像
# plt.show()

def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)

if __name__ == '__main__' :
    # t1 = np.arange(0, 5, 0.1)
    # t2 = np.arange(0, 5, 0.02)
    #
    # fig = plt.figure(12)
    # ax = plt.subplot(221)
    # plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')
    #
    # ax = plt.subplot(222)
    # plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
    #
    # plt.subplot(212)
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    #
    # plt.show()
    fig, ax1 = plt.subplots(1, 1)  # 做1*1个子图，等价于 " fig, ax1 = plt.subplot() "，等价于 " fig, ax1 = plt.subplots() "
    ax2 = ax1.twinx()

    # 作y=sin(x)函数
    x1 = np.linspace(0, 4 * np.pi, 100)
    y1 = np.sin(x1)
    ax1.plot(x1, y1)

    #  作y = cos(x)函数
    x2 = np.linspace(0, 4 * np.pi, 100)  # 表示在区间[0, 4π]之间取100个点（作为横坐标，“线段是有无数多个点组成的”）。
    y2 = np.cos(x2)
    ax2.plot(x2, y2, '*g')

    plt.show()