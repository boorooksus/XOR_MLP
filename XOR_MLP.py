"""
다층 신경망의 XOR 구현
"""

import numpy as np

# 초기화
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 입력 값
expected_output = np.array([[0], [1], [1], [0]])  # 실제 결과
predicted_output = np.array([])  # 예측 결과

layers = [2, 2, 1]  # [input layer, hidden layer, output layer]
epoch = 20000
learning_rate = 0.1


def sigmoid(x: np.array) -> np.array:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_der(x: np.array) -> np.array:
    return x * (1 - x)


def neural_network():
    global predicted_output

    # Random weights and bias initialization
    hidden_w = np.random.randn(layers[0], layers[1])  # weight
    hidden_b = np.random.randn(1, layers[1])  # bias
    output_w = np.random.randn(layers[1], layers[2])  # weight
    output_b = np.random.randn(1, layers[2])  # bias

    # Neural Networks
    for i in range(epoch):

        # Forward Propagation

        # input layer to hidden layer
        hidden_output = sigmoid(np.dot(inputs, hidden_w) + hidden_b)

        # hidden layer to output layer
        predicted_output = sigmoid(np.dot(hidden_output, output_w) + output_b)

        # Back propagation

        err = expected_output - predicted_output
        # output to hidden layer
        re_predicted_output = err * sigmoid_der(predicted_output)
        # hidden to input layer
        re_hidden_output = re_predicted_output.dot(output_w.T) * sigmoid_der(hidden_output)

        # update weights and biases
        output_w += learning_rate * hidden_output.T.dot(re_predicted_output)
        output_b += np.sum(re_predicted_output, axis=0, keepdims=True) * learning_rate

        hidden_w += learning_rate * inputs.T.dot(re_hidden_output)
        hidden_b += np.sum(re_hidden_output, axis=0, keepdims=True) * learning_rate

    return
    

def adjustment():
    for i, v in enumerate(predicted_output):
        v[0] = 1 if v[0] > 0.5 else 0
    return


if __name__ == '__main__':
    print("입력 데이터: ")
    print(inputs)

    print("\n예측 중...\n")
    neural_network()
    adjustment()
    
    print("예측 결과: ")
    print(predicted_output)
