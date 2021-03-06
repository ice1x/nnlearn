import os

import matplotlib.image as mpimg
import numpy as np  # linear algebra
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy.random as r
import matplotlib.pyplot as plt

# setup the NN structure
RESULT_SHAPE = 3
NN_STRUCTURE = [784, 300, RESULT_SHAPE]
ITER_NUM = 1000
PATH = os.path.abspath(os.path.curdir)
SHAPES = {
    0: 'triangles',
    1: 'circles',
    2: 'squares'
}


def convert_y_to_vect(y):
    """
    :param y:
    :return:
    """
    y_vect = np.zeros((len(y), RESULT_SHAPE))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


def f(x):
    return 1 / (1 + np.exp(-x))


def f_deriv(x):
    return f(x) * (1 - f(x))


def setup_and_init_weights():
    W = {}
    b = {}
    for l in range(1, len(NN_STRUCTURE)):
        W[l] = r.random_sample((NN_STRUCTURE[l], NN_STRUCTURE[l-1]))
        b[l] = r.random_sample((NN_STRUCTURE[l],))
    return W, b


def init_tri_values():
    tri_W = {}
    tri_b = {}
    for l in range(1, len(NN_STRUCTURE)):
        tri_W[l] = np.zeros((NN_STRUCTURE[l], NN_STRUCTURE[l-1]))
        tri_b[l] = np.zeros((NN_STRUCTURE[l],))
    return tri_W, tri_b


def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z


def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)


def train_nn(X, y, iter_num=ITER_NUM, alpha=0.25):
    W, b = setup_and_init_weights()
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values()
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(NN_STRUCTURE), 0, -1):
                if l == len(NN_STRUCTURE):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(NN_STRUCTURE) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y


def rgb2gray(img):
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    return 1 - img


if __name__ == "__main__":
    # load data and scale
    # digits = load_digits()
    files = []  # for storing all the images path
    result = []
    for index, shape in SHAPES.items():  # |
        new_path = os.path.join(PATH, shape)  # |
        for file_ in os.listdir(new_path):  # |How can I make this code shorter?
            files.append(os.path.join(new_path, file_))  # |
            result.append(index)

    images = []  # list for images
    for file_ in files:
        img = mpimg.imread(file_)
        img = rgb2gray(img)
        img = np.ravel(img)
        images.append(img)

    X_scale = StandardScaler()
    # X = X_scale.fit_transform(digits.data)
    # y = digits.target

    X = X_scale.fit_transform(images)
    y = result
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # convert digits to vectors
    y_v_train = convert_y_to_vect(y_train)
    y_v_test = convert_y_to_vect(y_test)

    # train the NN
    W, b, avg_cost_func = train_nn(X_train, y_v_train)
    # plot the avg_cost_func
    plt.plot(avg_cost_func)
    plt.ylabel('Average J')
    plt.xlabel('Iteration number')
    plt.show()
    # get the prediction accuracy and print
    y_pred = predict_y(W, b, X_test, 3)
    print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
