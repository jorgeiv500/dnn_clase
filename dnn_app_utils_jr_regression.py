
import numpy as np
import matplotlib.pyplot as plt
import h5py


# -------------------------
# Activaciones
# -------------------------

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


# -------------------------
# Costos
# -------------------------

def compute_cost(AL, Y):
    """Costo entropía cruzada (clasificación binaria)"""
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y, np.log(AL + 1e-8).T) - np.dot(1-Y, np.log(1-AL + 1e-8).T))
    return np.squeeze(cost)

def compute_mse(AL, Y):
    """Costo MSE (regresión)"""
    m = Y.shape[1]
    cost = (1./m) * np.sum((AL - Y)**2)
    return np.squeeze(cost)


# -------------------------
# Inicialización
# -------------------------

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


# -------------------------
# Forward
# -------------------------

def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    else:
        raise ValueError("activation debe ser 'sigmoid' o 'relu'")
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters, task="classification"):
    """task = 'classification' (última capa sigmoid) o 'regression' (última capa lineal)"""
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu"
        )
        caches.append(cache)

    if task == "classification":
        AL, cache = linear_activation_forward(
            A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid"
        )
    elif task == "regression":
        AL, cache = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
    else:
        raise ValueError("task debe ser 'classification' o 'regression'")

    caches.append(cache)
    return AL, caches


# -------------------------
# Backward
# -------------------------

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1./m) * np.dot(dZ, A_prev.T)
    db = (1./m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        raise ValueError("activation debe ser 'sigmoid' o 'relu'")
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, task="classification"):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    if task == "classification":
        dAL = - (np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] =             linear_activation_backward(dAL, current_cache, activation="sigmoid")
    elif task == "regression":
        dAL = (AL - Y)
        current_cache = caches[L-1]
        A_prev, W, b = current_cache
        m = A_prev.shape[1]
        grads["dW" + str(L)] = (1./m) * np.dot(dAL, A_prev.T)
        grads["db" + str(L)] = (1./m) * np.sum(dAL, axis=1, keepdims=True)
        grads["dA" + str(L-1)] = np.dot(W.T, dAL)
    else:
        raise ValueError("task debe ser 'classification' o 'regression'")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l+1)], current_cache, activation="relu"
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads


# -------------------------
# Update
# -------------------------

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters


# -------------------------
# Predicción y visualización
# -------------------------

def predict(X, y, parameters):
    probas, _ = L_model_forward(X, parameters, task="classification")
    p = (probas > 0.5).astype(int)
    print("Accuracy:", np.mean(p == y))
    return p

def print_mislabeled_images(classes, X, y, p):
    mislabeled_indices = np.asarray(np.where(p + y == 1))
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        plt.imshow(X[:, index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Pred: " + classes[int(p[0,index])].decode("utf-8") +
                  " / Real: " + classes[y[0,index]].decode("utf-8"))
        plt.show()

def plot_costs(costs):
    plt.plot(costs)
    plt.ylabel("Costo")
    plt.xlabel("Iteraciones")
    plt.title("Evolución del costo")
    plt.show()

def plot_predictions(y_true, y_pred, n=100):
    plt.figure(figsize=(8,5))
    plt.scatter(range(n), y_true[:n], label="Real", alpha=0.7)
    plt.scatter(range(n), y_pred[:n], label="Predicho", alpha=0.7)
    plt.legend()
    plt.title("Comparación Predicciones vs Reales")
    plt.show()
