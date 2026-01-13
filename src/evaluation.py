from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import math

def calculate_rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)
