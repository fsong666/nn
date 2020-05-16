import numpy as np
import copy
import random


class Network:
    def __init__(self, sizes=None, training_data=None, test_data=None, validation_data=None,
                 learning_rate=1.0, mini_batch_size=4, epochs=1):
        """
        一个例子sizes[2, 3, 3]　网络结构 2 * 3 * 3
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        # np.random.randn 标准高斯分布, 均值=0, 初始参数接近0,
        # w*x权重小相当于输入数据放大倍数小，抗数据波动性强，利于正则化
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for y, x in zip(sizes[1:], sizes[:-1])]

        self.training_data = training_data
        self.test_data = test_data
        self.validation_data = validation_data
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs

        self.learning_rate = learning_rate
        self.learning_rate_k = 0
        # 超过该迭代数，学习率不再衰减
        self.ites_threshold = len(training_data)
        # learning_rate_min = 最后值固定为0.01*learning_rate
        # learning_rate以斜率为(1-learning_rate_min)/ites_threshold,线性递减
        self.learning_rate_min = 0.1
        # 学习率的衰减率
        self.attenuation_rate = 0

        # bn
        self.bn_beta = [np.random.randn(y, 1) for y in sizes[1:]]
        self.bn_gamma = [np.random.randn(y, 1) for y in sizes[1:]]
        self.bn_min_const = [np.full(b.shape, pow(10, -8)) for b in self.biases]

        # moment
        self.moment_w = [np.zeros(w.shape) for w in self.weights]
        # alpha= 0.5, 0.9, 0.99
        self.moment_alpha = 0.5

        # RMSPro
        # learning_rate_Rho
        self.learning_rate_Rho = 0.7
        self.gradient_acc = [np.zeros(w.shape) for w in self.weights]

        # L2 Regularization [0.01...0.00001]
        self.L2Regular_alpha = 0.0001

        # early-stopping
        self.patience = 5000
        self.validation_frequency = min(len(self.training_data), int(self.patience / 2))

    def SGD(self):
        """
        全局损失降低＝＝每个样本的子损失减低的叠加 && 同一个模型的ｗ
        一次epoch的BatchGD == 一次epoch后的SGD
        一次epoch完整的数据对应完整的全局损失函数,
        SGD的每个样本的损失函数Loss_k(W[k-1]),W[k-1]:上个样本优化后的W
        Loss = 0.5(wx1 - y1)^2 + 0.5(wx2 - y2)^2 ...
        Loss1= 0.5(wx1 - y1)^2
        Gradient_w = (wx1 - y1)x1+ (wx2 - y1)x2 + ...
        Gradient1_w= (wx1 - y1)x1
        1.每个样本的损失函数是不同的，但数学表达形式相同.反向传播也只能求该样本对应的损失函数的梯度
        所以完整优化的是Full Btach 梯度下降(累积BP)，但为了减低计算成本，用mini_batchSGD
        在mini_batch的ＢＰ后的梯度是多个样本的均值，用局部近似全局损失函数的梯度
        2.虽然每个样本的损失函数是不同，但每次迭代，都是为了更新同一个模型的Ｗ，
        所以每个样本的损失函数BP减低一步，全局的损失函数也会相应降低一步.
        因为Full Btach的每个样本的损失之和==全局损失函数
        key:轮流依次降低每个样本的损失函数，且每次梯度是基于上个样本优化后的权重，
        所以每个样本的网络权重是基于上个样本优化后的权重进行更新,
        epoch最后一个样本更新后的，才完整的降低全局损失函数.
        3.所以每次bp前后的输入的样本值和每层的输入要保持不变，控制变量
        但网络是级联，会互相影响，引入BN去耦合，稳每层的输入
        """
        best_validation_accuracy = 0
        # 提高0.001倍, 提高self.patience
        improvement_threshold = 1.001
        # 连续３次验证acc降低，触发停止
        patience_increase = 5
        stop = False
        epoch = 0

        n = len(self.training_data)
        n_train_batches = int(n / self.mini_batch_size) + 1

        ites = 0
        # epoch[1,end]
        # 一次epoch完整的数据对应完整的全局损失函数,每个样本的损失函数是不同的，但形式相同
        # 一次epoch才是完整的全局损失函数优化
        while epoch < self.epochs and (not stop):
            epoch = epoch + 1
            random.shuffle(self.training_data)
            # mini_batch是列表中切割之后的列表
            mini_batches = [self.training_data[k:k + self.mini_batch_size]
                            for k in range(0, n, self.mini_batch_size)]
            mini_batch_index = -1
            for mini_batch in mini_batches:
                # mini_batch_index[0,end)
                mini_batch_index = mini_batch_index + 1
                self.update_mini_batch(mini_batch, ites)

                # ites[0, end)
                ites = (epoch - 1) * n_train_batches + mini_batch_index
                self.attenuation_rate = ites / self.ites_threshold

                if (ites + 1) % self.validation_frequency == 0:
                    this_validation_accuracy = self.evaluate(self.validation_data)

                    if ites <= self.ites_threshold + self.validation_frequency:
                        print("learning_rate_k: ", self.learning_rate_k)
                    print("iteration {0} accuracy: {1}".format(ites, this_validation_accuracy))

                    if this_validation_accuracy > best_validation_accuracy:
                        # improve patience if accuracy improvement is good enough
                        if this_validation_accuracy > best_validation_accuracy * improvement_threshold:
                            self.patience = max(self.patience, ites + patience_increase * self.validation_frequency)
                            print("patience increase:", self.patience)
                        best_validation_accuracy = this_validation_accuracy

                if ites >= self.patience:
                    stop = True
                    print("early-stop")
                    break
        print("Epoch {0} Test accuracy : {1}".format(epoch, self.evaluate(self.test_data)))
        # print('Epoch {0} Error Percent: {1}%'. \
        #       format(j, self.evaluate_regression(test_data) * 100))

    def update_mini_batch(self, mini_batch, ites):
        """
        更新一个mini_batch里的 w 和 b 的值
        """
        # 储存求和mini_batch里的每个样本对应的更新梯度 Partial derivative gradient
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # Nesterov-Moment 在前向计算梯度前先校正权重
        self.weights = [w + self.moment_alpha * moment for w, moment in zip(self.weights, self.moment_w)]

        for x, y in mini_batch:
            # x = x.reshape(-1, 1)
            zs, activations, maske_caches = self.forward(x)
            delta_nable_b, delta_nable_w = self.backprop(zs, activations, maske_caches, y)

            # mini_batch的梯度求和
            gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_nable_b)]
            gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_nable_w)]

        if ites <= self.ites_threshold:
            # learning_rate_k decrease to learning_rate_k*learning_rate_min(0.01)
            self.learning_rate_k = (1 - self.attenuation_rate) * self.learning_rate \
                                   + self.attenuation_rate * self.learning_rate * self.learning_rate_min

        # 每一个mini_batch更新一次参数,这是与sample by sample梯度下降最大区别
        # self.weights = [w - (self.learning_rate_k/ len(mini_batch)) * nw
        #                 for w, nw in zip(self.weights, gradient_w)]
        # (1 - self.learning_rate_k*self.alpha)*w: regularization
        # self.weights = [(1 - self.learning_rate_k * self.L2Regular_alpha) * w
        #                 - (self.learning_rate_k / len(mini_batch)) * nw
        #                 for w, nw in zip(self.weights, gradient_w)]
        weights = []
        moment = []
        # gradient_r = []
        for v, r, w, nw in zip(self.moment_w, self.gradient_acc, self.weights, gradient_w):
            # # # RMSPro
            # # r = self.learning_rate_Rho * r + (1 - self.learning_rate_Rho) * nw * nw
            # v = self.moment_alpha * v - (self.learning_rate_k / pow(r, 0.5)) * nw
            # Nesterov-Moment
            v = self.moment_alpha * v - (self.learning_rate_k / len(mini_batch)) * nw
            w = (1 - self.learning_rate_k * self.L2Regular_alpha) * w + v
            weights.append(w)
            moment.append(v)
            # gradient_r.append(r)
        self.weights = weights
        self.moment_w = moment
        # self.gradient_acc = gradient_r

        self.biases = [b - (self.learning_rate_k / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, gradient_b)]
        # self.bn_gamma = [g - self.learning_rate_k * gradient_g
        #                  for g, gradient_g in zip(self.bn_gamma, bn_gradient_gamma)]

    def forward(self, x):
        """
        输入是一个mini_batch里的每个样本，样本是轮流依次输入前向计算
        """
        activation = x
        activations = [x]
        zs = []
        maske_caches = []
        layer = 0
        for i in range(len(self.biases)):
            layer = layer + 1
            z = np.dot(self.weights[i], activation) + self.biases[i]
            zs.append(z)
            activation = self.sigmoid(z)

            # dropout, 输入层和最后一层不dropout
            maske = np.random.rand(activation.shape[0], 1)
            if layer == self.num_layers:
                maske = maske >= 0  # 全1
            else:
                maske = maske > 0.5  # 0.5的概率置１
                # 比如一个神经元的输出是x，那么在训练的时候它有p的概率参与训练，
                # (1-p)的概率丢弃，那么它输出的期望是px+(1-p)0=px,所以测试的前向阶段w*p得到相同期望
                # 补偿缩放，在测试的前向阶段w*p,or 在BP时激活函数值y: y/(1 - p)
                # 在测试的前向阶段w * 0.5 or 在BP时 y / (1 - 0.5) = y * 2 权重比例推断规则
                activation = activation * maske * 2
            maske_caches.append(maske)
            activations.append(activation)
        return zs, activations, maske_caches

    def backprop(self, zs, activations, maske_caches, y):
        """
        只反向求解梯度
        每次反向，只会优化当前样本对应的损失函数
        """
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        # 求损失函数对每层输入的梯度,即求 δ 的值
        # 下面求损失函数对输出层的输入的梯度,l = -1
        delta_node = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        gradient_b[-1] = delta_node
        gradient_w[-1] = np.dot(delta_node, activations[-2].T)
        # print("delta[-1] = \n", delta)

        for l in range(2, self.num_layers):
            # 下面这里利用 l+1 层的 δ 值来计算 l 的 δ 值
            z = zs[-l]
            maske_bp = maske_caches[-l]
            # sp = self.sigmoid_prime(activations[-l])
            # δ(-l) = W(-l+1).T dot δ(-l+1) * f'(z(-l))
            # delta作为输入,　为了W矩阵的行数==delta的列数,Ｗ要转置
            delta_node = maske_bp * np.dot(self.weights[-l + 1].T, delta_node) * self.sigmoid_prime(z)
            # delta_node = np.dot(self.weights[-l + 1].T, delta_node) * self.sigmoid_prime(z)
            # print("delta[{0}] = \n {1} ".format(-l, delta))
            gradient_b[-l] = delta_node
            # delta_w(-l) =  δ(-l) dot a(-l-1).T
            gradient_w[-l] = np.dot(delta_node, activations[-l - 1].T)
        return gradient_b, gradient_w

    def sigmoid(self, z):  # sigmoid方程
        # z很小时exp(-z)极大值，导致overflow, 相对exp(z)是极小值，不会overflow
        # OverflowError::res = 1.0 / (1.0 + np.exp(-z))
        return 0.5 * (1 + np.tanh(0.5 * z))

    def sigmoid_prime(self, z):  # sigmoid方程的求导
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cost_derivative(self, out, y_label):
        return out - y_label

    def forward_single(self, x):
        for b, w in zip(self.biases, self.weights):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    def forward_multi(self, mini_batch):
        activations = [x for x in mini_batch]
        z = copy.deepcopy(activations)
        for i in range(len(self.biases)):
            for m in range(self.mini_batch_size):
                z[m] = np.dot(self.weights[i], activations[m]) + self.biases[i]
                activations = self.sigmoid(z[m])
        return activations

    def forward_bn(self, mini_batch):
        """
        一个mini_batch的多个样本，同时一起输入到每层，可求得每层激活函数输出的均值
        """
        activations = [x for x in mini_batch]
        z = copy.deepcopy(activations)
        z_bn = copy.deepcopy(activations)
        z_mean = [np.zeros(b.shape) for b in self.biases]
        z_var = [np.zeros(b.shape) for b in self.biases]
        z_std = [np.zeros(b.shape) for b in self.biases]

        list_activations = []
        list_z = []
        list_z_bn = []

        for i in range(len(self.biases)):
            for m in range(self.mini_batch_size):
                z[m] = np.dot(self.weights[i], activations[m]) + self.biases[i]
                z_mean[i] = z_mean[i] + z[m]
            list_z.append(z)
            z_mean[i] = z_mean[i] / self.mini_batch_size
            for m in range(self.mini_batch_size):
                z_var[i] = z_var[i] + (z_mean[i] - z[m]) * (z_mean[i] - z[m])
            z_var[i] = z_var[i] / self.mini_batch_size
            z_std[i] = pow(z_var[i] + self.bn_min_const[i], 0.5)
            for m in range(self.mini_batch_size):
                z_bn[m] = (z[m] - z_mean[i]) / z_std[i]
                activations[m] = self.sigmoid(z_bn[m])
            list_z_bn.append(z_bn)
            list_activations.append(activations)
        return activations, list_z, list_z_bn, list_activations

    def forward_bn_test(self, x, mean, var):
        """
        用于带bn的测试
        测试时，没有，也无需mini_batch，所以轮流依次前向计算
        """
        activation = x
        for i in range(len(self.biases)):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            z = (z - mean[i]) / pow(var[i] + self.bn_min_const[i], 0.5)
            z = self.bn_gamma[i] * z + self.bn_beta[i]
            activation = self.sigmoid(z)
        return activation

    def evaluate_bn(self, test_data, mean, var):
        test_results = [(np.argmax(self.forward_bn_test(x, mean, var)), y)
                        for (x, y) in test_data]
        sum_value = sum(int(x == y) for (x, y) in test_results)
        return sum_value / len(test_data)

    def evaluate(self, test_data):
        # np.argmax()返回一个多维数组值最大的索引值,索引是一维索引，索引值是个标量
        test_results = [(np.argmax(self.forward_single(x)), y)
                        for (x, y) in test_data]
        # 对一个比对结果的list求和
        sum_value = sum(int(x == y) for (x, y) in test_results)
        return sum_value / len(test_data)

    def evaluate_regression(self, test_data):
        y_pre = []
        diff = []
        y_label = []
        X = []
        for x, y in test_data:
            X.append(x.item())
            yy = self.forward_single(x.reshape(-1, 1)).item()
            y_pre.append(yy)
            y_label.append(y.item())
            diff.append(abs(yy - y).item())
        # print('X = \n', X)
        # print('y_label = \n', y_label)
        # print('y_pre = \n', y_pre)
        # print('diff = \n', diff)
        #
        # plt.scatter(X, y_label, c='g', marker='o')
        # plt.scatter(X, y_pre, c='r', marker='*')
        # plt.show()
        test_results = sum(diff) / len(test_data)
        return test_results
