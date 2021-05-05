import numpy as np
import matplotlib.pyplot as plt


def feature_normalize(data_set):
    feature_normalized = list()
    mean = np.mean(data_set)
    standard_deviation = np.std(data_set)
    for data in data_set:
        fnd = (data - mean) / standard_deviation  #feature normalized data
        feature_normalized.append(fnd)

    return feature_normalized, mean, standard_deviation


def lr_gradient_descent(features, correct_answers, features_count, training_set_size, iterations, alpha):
    """
    linear regression with multiple variables
    :param features: X
    :param correct_answers: y
    :param features_count: n
    :param training_set_size: m
    :param iterations: the number of times to iterate
    :param alpha: learning rate
    :return: parameters
    """
    Theta = np.zeros(features_count)
    for _ in range(iterations):
        gd_sum = np.zeros(features_count)
        for i in range(features_count):
            for j in range(training_set_size):
                gd_sum[i] += (np.matmul(
                    Theta,
                    np.matrix.transpose(features[j])
                ) - correct_answers[j]) * features[j][i]
        for k in range(features_count):
            Theta[k] = Theta[k] - (alpha * gd_sum[k] / training_set_size)

    return Theta


def model_data(path, features_and_target_names, iterations, alpha):
    training_data = np.genfromtxt(
        path,
        delimiter=",",
        names=features_and_target_names
    )
    data_sets = [training_data[name] for name in features_and_target_names]
    m = len(data_sets[0])  # training set size
    fn_data_sets = list()  # feature normalized data sets
    for data_set in data_sets:
        fn_data_sets.append(
            feature_normalize(data_set)
        )


    length = len(features_and_target_names)
    data_sets_fn = [fn_data_sets[i][0] for i in range(length)]
    features_fn = data_sets_fn[:-1]
    y = data_sets_fn[-1]
    X = np.vstack((np.ones(m), *features_fn)).T
    features_count = len(X[0])
    return X, y, features_count, m, iterations, alpha


(X, y, features_count, m, iterations, alpha) = model_data(
    path="ex1data2.txt",
    features_and_target_names=["size", "bedroom", "price"],
    iterations=5500,
    alpha=0.001
)
Theta = lr_gradient_descent(
    features=X,
    features_count=features_count,
    correct_answers=y,
    training_set_size=m,
    iterations=iterations,
    alpha=alpha
)
print(Theta)

