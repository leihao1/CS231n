from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    C = W.shape[1]
    for i in range(N):
        scores = X[i]@W
        target = y[i]
        loss -= scores[target]
        total = 0.0
        for j in range(C):
            total += np.exp(scores[j])
        loss += np.log(total)
        
        dW[:,target] += -X[i] + (np.exp(scores[target])*X[i])/total
        for j in range(C):
            if j != target:
                dW[:,j] += (np.exp(scores[j])*X[i])/total
    loss /= N
    loss += reg * np.sum(W * W)
    
    dW /= N
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X@W
    scores -= np.max(scores)
    targets = scores[np.arange(y.size),y]
    exp = np.exp(scores)
    total = np.sum(exp,axis=1)
    loss += np.mean(np.log(total)-targets) + reg*np.sum(W*W)
    
    mask = np.zeros_like(scores)
    mask[np.arange(y.size),y] = 1
    dW = X.T@(exp.T / total).T - X.T@mask
    dW /= X.shape[0]
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
