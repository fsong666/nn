
def cnnSize(input, kernel_size, stride=1, padding=0):
    y = (input - kernel_size + 2 * padding) // stride + 1
    print('out_cnnSize: ', y)
    return y


def maxPoolSize(input, kernel_size, stride=1, padding=0, dilation=1):
    y = (input + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    print('out_MaxPoolSize: ', y)
    return y


if __name__ == '__main__':
    x = cnnSize(input=32, kernel_size=5, stride=1, padding=2)
    x = maxPoolSize(input=x, kernel_size=2, stride=2, padding=0)
    x = cnnSize(input=x, kernel_size=5, stride=1, padding=2)
    x = maxPoolSize(input=x, kernel_size=2, stride=2, padding=0)
    x = cnnSize(input=x, kernel_size=5, stride=1, padding=2)
    x = maxPoolSize(input=x, kernel_size=2, stride=2, padding=0)
