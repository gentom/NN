from math import exp
import forward_propagation as fp


# Make a prediction with a network
def predict(network, row):
    outputs = fp.forward_propagate(network, row)
    return outputs.index(max(outputs))


if __name__ == '__main__':
    # Test making predictions with the network
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]
    network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
               [{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
    for row in dataset:
        prediction = predict(network, row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))
