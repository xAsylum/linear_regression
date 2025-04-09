import numpy as np


def gauss(x, s):
    return np.exp(-((x / s) ** 2))


def f(X):
    coefs = [549.35301, 136.7966, -9.11111, -6.65216, 459.217, 155.59317, 0.45358, 9.11111, 9.06219, -1.14926, 89.03875, -3.1117, -0.417, 0.59496, 1.14926, 7.061, 0.99701, 0.9832, -2.27029, 0.61326, -7.061, 34.7227, 1.59305, -0.69634, 0.58788, -0.99701, 141.33767, 103.44053, -0.67209, -0.9832, 19.95783, 0.0867, 2.27029, 15.96881, -0.61326, 7.061, -0.0, 3.68088, -1.58336, 0.0, -37.84482, -0.0, 68.94492, 385.38169, 52.33111, 90.82989, 64.4988, 60.28962]
    mean = [np.float64(-0.3986655546288574), np.float64(-0.6989157631359466), np.float64(4.48790658882402), np.float64(18.11759799833194), np.float64(44.84987489574645), np.float64(-1.005004170141785), np.float64(12.397831526271894)]
    std = [np.float64(5.763871679234808), np.float64(2.8534263953894072), np.float64(1.708921457646692), np.float64(12.008907546160872), np.float64(8.646856153277609), np.float64(1.4206771561950318), np.float64(5.7068527907788145)]
    res = []
    for y in X:
        x = y
        x = [(x[i] - mean[i])/std[i] for i in range(len(mean))]
        x = [1, *x]
        func = [*[x[i] for i in range(8)],
            *[x[i] * x[j] for i in range(1, 8) for j in range(i, 8)],
            gauss(x[1], 0.01), gauss(x[4], 0.01), gauss(x[5], 0.01), gauss(x[3], 0.01), gauss(x[6], 0.01),
            gauss(x[7], 0.01),
            gauss(x[1], 0.5), gauss(x[4], 0.5), gauss(x[5], 0.5), gauss(x[3], 0.5), gauss(x[6], 0.5),
            gauss(x[7], 0.5)]
        prediction = np.dot(coefs, func)
        res.append(prediction)
    res = np.array(res)
    return res


data = np.loadtxt('dane.data')

X = data[:, 0:7]
y = data[:, -1]

predictions = f(X)
error = predictions - y
SSE = np.sum(error ** 2)
RMSE = np.sqrt(SSE / len(y))
print(RMSE)
