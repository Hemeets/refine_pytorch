'''
Author: QDX
Date: 2023-04-17 09:17:28
Description: 
'''
import torch as t
import random
from d2l import torch as d2l
from matplotlib import pyplot as plt


def synthetic_data(w, b, num_examples):
    ''' 生成假数据 '''
    X = t.normal(0, 1, (num_examples, len(w)))
    y = t.matmul(X, w) + b
    y += t.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = t.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]


# def show_synthetic_data():
#     ''' 通过⽣成第⼆个特征features[:, 1]和labels的散点图，可以直观观察到两者之间的线性关系 '''
#     true_w = t.tensor([2, -3.4])
#     true_b = 4.2
#     features, labels = synthetic_data(true_w, true_b, 1000)
#     print('features:', features[0],'\nlabel:', labels[0])
#     d2l.set_figsize()
#     d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
#     plt.show()

#     batch_size = 10
#     for X, y in data_iter(batch_size, features, labels):
#         print(X, '\n', y)
#         break


def linreg(x, w, b):
    """ linear regressive model """
    return t.matmul(x, w) + b


def squared_loss(y_hat, y):
    """ mean squared error, loss function """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """ batch stochastic gradient descent """
    with t.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def run_model():

    # show_synthetic_data
    true_w = t.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    # 通过⽣成第⼆个特征features[:, 1]和labels的散点图，可以直观观察到两者之间的线性关系
    print('features:', features[0],'\nlabel:', labels[0])
    d2l.set_figsize()
    d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    plt.show()

    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

    ## train
    # initial
    w = t.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = t.zeros(1, requires_grad=True)

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for x, y in data_iter(batch_size, features, labels):
            l = loss(net(x, w, b), y)
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with t.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')



if __name__ == "__main__":
    # show_synthetic_data()
    run_model()