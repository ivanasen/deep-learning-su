import numpy as np
from random import shuffle
from past.builtins import xrange
from functools import reduce


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        correct_class_score = scores[y[i]]

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        loss += -np.log(probs[y[i]])

        probs[y[i]] -= 1
        dW += np.outer(probs, X[i]).T
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = X.shape[1]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1)[:, np.newaxis]

    correct_scores = np.choose(y, scores.T)

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1)[:, np.newaxis]
    y_probs = np.choose(y, probs.T)

    loss = np.sum(-np.log(y_probs)) / num_train + reg * np.sum(W * W)

    probs[range(num_train), y] -= 1
    probs /= num_train
    dW = np.dot(X.T, probs)
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
