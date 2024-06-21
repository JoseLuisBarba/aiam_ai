import numpy as np

def MSE(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    y, y_hat = np.array(y), np.array(y_hat)
    diff = np.subtract(y, y_hat)
    squared_diff = np.square(diff)
    return np.mean(squared_diff)


def RMSE(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    y, y_hat = np.array(y), np.array(y_hat)
    diff = np.subtract(y, y_hat)
    squared_diff = np.square(diff)
    mean_squared_diff = np.mean(squared_diff)
    return np.sqrt(mean_squared_diff)


def MAE(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    y, y_hat = np.array(y), np.array(y_hat)
    diff = np.subtract(y, y_hat)
    absolute_diff = np.absolute(diff)
    return np.mean(absolute_diff) 


def MAPE(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    y, y_hat = np.array(y), np.array(y_hat)
    diff = np.subtract(y, y_hat)
    percental_diff = np.divide(np.absolute(diff), y_hat+0.01)
    absolute_diff = np.absolute(percental_diff)
    return np.mean(absolute_diff) * 100

def huber_loss(y: np.ndarray, y_hat: np.ndarray, delta: float=1.35):
    y, y_hat = np.array(y), np.array(y_hat)
    diff = np.subtract(y, y_hat)
    huber_mse = 0.5 * np.square(diff)
    huber_mae = delta * (np.absolute(diff) - 0.5*delta)
    return np.mean(
        np.where(np.absolute(diff) <= delta,
            huber_mse,
            huber_mae
        )
    )


def categorical_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    y, y_hat = np.array(y), np.array(y_hat)
    ce = -np.sum(np.multiply(y, np.log10(y_hat)))
    return ce


def hinge_loss(y: np.ndarray, y_hat: np.ndarray):
    y, y_hat = np.array(y), np.array(y_hat)
    return np.max(0, 1 - y*y_hat)
