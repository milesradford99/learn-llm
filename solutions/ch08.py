"""Reference solutions for chapter 08. Peek only after trying!"""

import numpy as np


def softmax(S):
    S = S - np.max(S, axis=1, keepdims=True)
    E = np.exp(S)
    return E / np.sum(E, axis=1, keepdims=True)


def relu(Z):
    return np.maximum(Z, 0)


def forward(X, W1, b1, W2, b2):
    Z = X @ W1 + b1
    H = relu(Z)
    S = H @ W2 + b2
    return Z, H, S


def backward(X, Y, Z, H, S, W2):
    G = (softmax(S) - Y) / len(X)
    dW2 = H.T @ G
    db2 = np.sum(G, axis=0)
    dH = G @ W2.T
    dZ = dH * (Z > 0)
    dW1 = X.T @ dZ
    db1 = np.sum(dZ, axis=0)
    return dW1, db1, dW2, db2
