from collections import defaultdict
import numpy as np
def load_data(path):
    with open(path, "r") as f:
        records = f.readlines()
        samples = [record.split(",") for record in records]
        inputs = [sample[0:4] for sample in samples]
        labels = [sample[4][0:-1] for sample in samples]

    return inputs, labels

def p_classes(table, labels):
    classes = defaultdict(int)
    for label in labels:
        classes[label] += 1

    for c in classes:
        table[c]["p_class"] = classes[c] / len(labels)

    return table

def mean_variance(table, inputs, labels):
    classes = dict()
    classes_name = list(set(labels))
    features_size = len(inputs[0])

    for name in classes_name:
        classes[name] = {}
        for num in range(features_size):
            classes[name][str(num)] = [] 

    for i in range(len(inputs)):
        for j in range(features_size):
            classes[labels[i]][str(j)].append(float(inputs[i][j]))

    for c in classes:
        for feature in classes[c]:
            table[c][feature] = [np.mean(classes[c][feature]), np.var(classes[c][feature])]
    return table


def probability_density_function(x, mean, variance, epsilon=1e-6):
    pd = 1 / (np.sqrt(2 * np.pi * (variance  + epsilon))) * np.exp((-(x - mean)**2) / (2 * variance + epsilon))
    return pd

def create_table(labels):
    table = {}
    labels = list(set(labels))
    for label in labels:
        table[label] = {}

    return table

def predict(x, table):
    for c in table:
        p_class_features = table[c]["p_class"]
        for feature in range(len(x)):
            mean, var = table[c][str(feature)]
            p_class_features *= probability_density_function(x[feature], mean, var)

        print(f"class {c}: {p_class_features}")

def naive_bayes():
    inputs, labels = load_data("./iris.data")
    table = create_table(labels)
    table = p_classes(table, labels)
    table = mean_variance(table, inputs, labels)

    return table

def main():
    table = naive_bayes()
    x = [5.4,3.9,1.3,0.4]
    predict(x, table)


if __name__=="__main__":
    main()