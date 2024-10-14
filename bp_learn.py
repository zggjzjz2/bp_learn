import numpy as np

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return (z>0).astype(float)

def sigmoid(z):
    return 1/(1+np.exp(-z))

# 输入数据
x = np.array([1.0, 2.0])
y = 0  # 真实输出

# 初始化权重和偏置
W1 = np.array([[0.2, -0.4],
               [0.7, 0.1]])
b1 = np.array([0.1, -0.2])

W2 = np.array([0.6, -0.1])
b2 = 0.2

# 前向传播
z1 = np.dot(W1,x)+b1
a1 = relu(z1)
z2 = np.dot(W2,a1)+b2
a2 = sigmoid(z2)

# 计算损失（交叉熵损失）
loss = - (y * np.log(a2) + (1 - y) * np.log(1 - a2))
print(f"损失: {loss}")

# 反向传播
# 输出层梯度
dz2 = a2 - y  # ∂L/∂z2
dW2 = dz2 * a1  # ∂L/∂W2
db2 = dz2  # ∂L/∂b2

# 隐藏层梯度
da1 = W2 * dz2  # ∂L/∂a1
dz1 = da1 * relu_derivative(z1)  # ∂L/∂z1
dW1 = np.outer(dz1, x)  # ∂L/∂W1
db1 = dz1  # ∂L/∂b1

# 更新参数（学习率为0.1）
learning_rate = 0.1
W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2

# 更新后的参数
print("更新后的权重和偏置：")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)
#更改，再改
