import math
import random

def naive_softmax(X):
    X_exp = [math.exp(x) for x in X]
    x_exp_sum = sum(X_exp, 0.0)
    X_softmax = [xexp / x_exp_sum for xexp in X_exp]
    
    return X_softmax

def safe_softmax(X):
    epsilon = 0
    m = [-math.inf for _ in X]

    m[0] = X[0]
    for k, xk in enumerate(X[1:]):
        k += 1
        m[k] = max(m[k-1], xk)

    d = [0 for _ in X]

    d[0] = math.exp(X[0] - m[-1])
    for j, xj in enumerate(X[1:]):
        j += 1
        d[j] = d[j-1] + math.exp(xj - m[-1])

    y = [0 for _ in X]

    for i, xi in enumerate(X):
        y[i] = math.exp(xi - m[-1]) / d[-1]

    return y

def online_softmax(X):
    m = [-math.inf for _ in X]
    d = [0 for _ in X]
    y = [0 for _ in X]

    m[0] = X[0]
    d[0] = 1
    for j, xj in enumerate(X[1:]):
        j += 1
        m[j] = max(m[j-1], xj)
        d[j] = d[j-1] * math.exp(m[j-1] - m[j]) + math.exp(xj - m[j])

    for i, xi in enumerate(X):
        y[i] = math.exp(xi - m[-1]) / d[-1]

    return y

# X = [ random.randint(1, 5000) for _ in range(10)]
X = [ 2 * n for n in reversed(range(10))]

print(f"X = {X}")
try:
    print(f"naive_softmax(X) {naive_softmax(X)}")
except:
    print("naive boi being naive")

print(f"safe_softmax(X) {safe_softmax(X)}")
print(f"online_softmax(X) {online_softmax(X)}")
