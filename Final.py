# Author: Tuan Nguyen
# Date: 2019-04-23
from math import sqrt
from numpy import array
from scipy.optimize import fmin_l_bfgs_b
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


data = []
date = []
# Open file
with open("list.csv") as f:
    for row in f:
        data.append(int(row.strip().split(',')[1]))
        date.append(row.split(',')[0])
# Close file
f.close()


def RMSE(parameters, *arguments):
    # Purpose:
    # This function takes arguments such as historic data, prediction model and
    # period want to predict
    # Signature:
    # RMSE :: (parameters: (Float, Float, Float),
    # arguments: (List, String, Integer)) => Float
    # :return: Float
    X = arguments[0]
    model = arguments[1]
    rmse = 0
    if model == "linear_exponential":
        # Get data from parameters
        alpha, beta = parameters
        # Initialization
        u = [X[0]]
        v = [X[1] - X[0]]
        y = [u[0] + v[0]]
        # Apply linear exponential model formula
        for i in range(len(X)):
            u.append(alpha * X[i] + (1 - alpha) * (u[i] + v[i]))
            v.append(beta * (u[i + 1] - u[i]) + (1 - beta) * v[i])
            y.append(u[i + 1] + v[i + 1])
    else:
        # Get data from parameters
        alpha, beta, gamma = parameters
        m = arguments[2]
        # Initialization
        u = [sum(X[0:m]) / float(m)]
        v = [(sum(X[m:2 * m]) - sum(X[0:m])) / m ** 2]
        if model == "additive":
            # Apply additive model formula
            s = [X[i] - u[0] for i in range(m)]
            y = [u[0] + v[0] + s[0]]
            for i in range(len(X)):
                u.append(alpha * (X[i] - s[i]) + (1 - alpha) * (u[i] + v[i]))
                v.append(beta * (u[i + 1] - u[i]) + (1 - beta) * v[i])
                s.append(gamma * (X[i] - u[i] - v[i]) + (1 - gamma) * s[i])
                y.append(u[i + 1] + v[i + 1] + s[i + 1])
        elif model == "multiplicative":
            # Apply multiplicative model formula
            s = [X[i] / u[0] for i in range(m)]
            y = [(u[0] + v[0]) * s[0]]
            for i in range(len(X)):
                u.append(alpha * (X[i] / s[i]) + (1 - alpha) * (u[i] + v[i]))
                v.append(beta * (u[i + 1] - u[i]) + (1 - beta) * v[i])
                s.append(gamma * (X[i] / (u[i] + v[i])) + (1 - gamma) * s[i])
                y.append((u[i + 1] + v[i + 1]) * s[i + 1])
    # Calculate rmse value
    rmse = sqrt(sum([(i - j) ** 2 for i, j in zip(X, y[:-1])]) / len(X))
    return rmse


def multiplicative(x, m, fc, confidence, best=None):
    # Purpose:
    # This function takes historic data, period, prediction period, confidence,
    # alpha, beta, gamma, and best. If best is none, return rsme, else, pass values to
    # result function
    # Signature:
    # multiplicative :: ((List, Integer, Integer, Float, Integer))
    # => Float
    # :return: Float
    # x = historic data values in array backed list
    # m = periodicity of the data
    # fc = number of period to forecast into the future
    X = x[:]
    his_data = draw_his(X)
    # Initialization
    initial_vals = array([0.0, 1.0, 0.0])
    limits = [(0, 1), (0, 1), (0, 1)]
    model = "multiplicative"
    parameters = fmin_l_bfgs_b(RMSE, x0=initial_vals,
                               args=(X, model, m), bounds=limits, approx_grad=True)
    alpha, beta, gamma = parameters[0]
    # Initialization
    u = [sum(X[0:m]) / float(m)]
    v = [(sum(X[m:2 * m]) - sum(X[0:m])) / m ** 2]
    s = [X[i] / u[0] for i in range(m)]
    y = [(u[0] + v[0]) * s[0]]
    rmse = 0
    counter = len(X)
    # Apply multiplicative formula
    for i in range(len(X) + fc):
        if i == len(X):
            X.append((u[-1] + v[-1]) * s[-m])
        u.append(alpha * (X[i] / s[i]) + (1 - alpha) * (u[i] + v[i]))
        v.append(beta * (u[i + 1] - u[i]) + (1 - beta) * v[i])
        s.append(gamma * (X[i] / (u[i] + v[i])) + (1 - gamma) * s[i])
        y.append((u[i + 1] + v[i + 1]) * s[i + 1])
    # Calculate rmse value
    rmse = sqrt(sum([(i - j) ** 2 for i, j in zip(X[:-fc], y[:-fc - 1])]) / len(X[:-fc]))
    if best is None:
        return rmse
    else:
        print("Forecasted Demand from Optimized Multiplicative Model")
        result(X, y, m, fc, confidence, his_data, counter, rmse)


def additive(x, m, fc, confidence, best=None):
    # Purpose:
    # This function takes historic data, period, prediction period, confidence,
    # alpha, beta, gamma, and best. If best is none, return rsme, else, pass values to
    # result function
    # Signature:
    # additive :: ((List, Integer, Integer, Float, Integer)) => Float
    # :return: Float
    # x = historic data values in array backed list
    # m = periodicity of the data
    # fc = number of period to forecast into the future
    X = x[:]
    his_data = draw_his(X)
    # Initialization
    initial_vals = array([0.0, 1.0, 0.0])
    limits = [(0, 1), (0, 1), (0, 1)]
    model = "additive"
    parameters = fmin_l_bfgs_b(RMSE, x0=initial_vals,
                               args=(X, model, m), bounds=limits, approx_grad=True)
    alpha, beta, gamma = parameters[0]
    # Initialization
    u = [sum(X[0:m]) / float(m)]
    v = [(sum(X[m:2 * m]) - sum(X[0:m])) / m ** 2]
    s = [X[i] - u[0] for i in range(m)]
    y = [u[0] + v[0] + s[0]]
    rmse = 0
    counter = len(X)
    # Apply additive model formula
    for i in range(len(X) + fc):
        if i == len(X):
            X.append(u[-1] + v[-1] + s[-m])
        u.append(alpha * (X[i] - s[i]) + (1 - alpha) * (u[i] + v[i]))
        v.append(beta * (u[i + 1] - u[i]) + (1 - beta) * v[i])
        s.append(gamma * (X[i] - u[i] - v[i]) + (1 - gamma) * s[i])
        y.append(u[i + 1] + v[i + 1] + s[i + 1])
    # Calculate rmse value
    rmse = sqrt(sum([(i - j) ** 2 for i, j in zip(X[:-fc], y[:-fc - 1])]) / len(X[:-fc]))
    if best is None:
        return rmse
    else:
        print("Forecasted Demand from Optimized Additive Model")
        result(X, y, m, fc, confidence, his_data, counter, rmse)


def linear_exponential(x, p, fc, confidence, best=None):
    # Purpose:
    # This function takes historic data, prediction period, confidence,
    # alpha, beta, and best. If best is none, return rsme, else, pass values to
    # result function
    # Signature:
    # linear_exponential :: ((List, Integer, Integer, Integer)) => Float
    # :return: Float
    # x = historic data values in array backed list
    # fc = number of period to forecast into the future
    X = x[:]
    his_data = draw_his(X)
    # Initialization
    initial_vals = array([0.0, 1.0])
    limits = [(0, 1), (0, 1)]
    model = "linear_exponential"
    parameters = fmin_l_bfgs_b(RMSE, x0=initial_vals,
                               args=(X, model), bounds=limits, approx_grad=True)
    alpha, beta = parameters[0]
    # Initialization
    u = [X[0]]
    v = [X[1] - X[0]]
    y = [u[0] + v[0]]
    rmse = 0
    counter = len(X)
    # Apply linear exponential formula
    for i in range(len(X) + fc):
        if i == len(X):
            X.append(u[-1] + v[-1])
        u.append(alpha * X[i] + (1 - alpha) * (u[i] + v[i]))
        v.append(beta * (u[i + 1] - u[i]) + (1 - beta) * v[i])
        y.append(u[i + 1] + v[i + 1])
    # Calculate rmse value
    rmse = sqrt(sum([(i - j) ** 2 for i, j in zip(X[:-fc], y[:-fc - 1])]) / len(X[:-fc]))
    if best is None:
        return rmse
    else:
        print("Forecasted Demand from Linear exponential Model")
        result(X, y, p, fc, confidence, his_data, counter, rmse)


def draw_his(X):
    # Purpose: This function appends the increment period of
    # the historic data from csv file to the list and
    # then plot the data out from the list
    # Signature:
    # draw_his :: (List) => List
    # :return: List
    his_data = []
    for i in range(1, len(X) + 1):
        his_data.append(i)
    # if statement to make sure the test data below not print out
    if len(his_data) > 50:
        plt.plot(his_data, X, label="Data")
    return his_data


def result(X, y, p, fc, confidence, his_data, counter, rmse):
    # Purpose:
    # This function takes historic data, prediction data, data period, prediction
    # period, confidence, historic data plot, length of historic data, rmse
    # Signature:
    # result :: ((List, List, Integer, Integer, Float, List, Integer, Float)) => List
    date_string = str(date[-1])
    date_list = date_string.split('/')
    period1 = int(date_list[0])
    year = int(date_list[1])
    res = 0
    residuals = []
    dupper = []
    dlower = []
    periodl = []
    predictl = []
    cf = abs(scipy.stats.norm.ppf((1 - confidence) / 2))
    if p == 12:
        first_year_period = 12
    elif p == 4:
        first_year_period = 4
    # Calculate residuals which is the error of the prediction and append them
    # to the list
    for m, n in zip(X[first_year_period:-fc], y[first_year_period:-fc - 1]):
        res = m - n
        residuals.append(res)
    # Print out all the prediction data depends on what the prediction period that
    # user selected
    for x in (X[-fc:]):
        predict = int(round(x))
        upper = int(round(x + cf * rmse))
        lower = int(round(x - cf * rmse))
        counter += 1
        if p == 12:
            period1 += 1
            if (period1 > 12):
                period1 = 1
                year += 1
        elif p == 4:
            period1 += 1
            if (period1 > 4):
                period1 = 1
                year += 1
        predictl.append(predict)
        dupper.append(upper)
        dlower.append(lower)
        periodl.append(counter)
        # Print out the result
        print("| " + str('%02d' % period1) + "/" + str('%02d' % year) + " | "
              + str(predict).ljust(5) + " | " + str(lower).ljust(5)
              + ' - ' + str(upper).ljust(5) + " | ")
    # Plot all the data to the graph
    month_predict = his_data[first_year_period:] + periodl
    tot_predict = y[first_year_period:-fc - 1] + predictl
    plt.bar(his_data[first_year_period:], residuals, 0.5, label="Residuals")
    plt.plot(month_predict, tot_predict, label="Predict")
    plt.fill_between(periodl, dupper, dlower, color='grey', alpha='0.5')


def best(period, prediction_period, confidence):
    # Purpose:
    # This function takes rmse from three prediction models above and get the smallest
    # rsme from three models and select the best model and call the function
    # of that model to print out the result
    # Signature:
    # result :: ((Integer, Integer, Float))
    if period == "monthly":
        period = 12
    elif period == "quarterly":
        period = 4
    else:
        exit("Period must be monthly or quarterly")
    Multi_rmse = multiplicative(data, period, prediction_period, confidence)
    print("Optimized Multiplicative's RMSE:", Multi_rmse)
    Add_rmse = additive(data, period, prediction_period, confidence)
    print("Optimized Additive's RMSE:", Add_rmse)
    Linear_rmse = linear_exponential(data, period, prediction_period, confidence)
    print("Linear exponential's RMSE:", Linear_rmse)
    if min(Multi_rmse, Add_rmse, Linear_rmse) == Linear_rmse:
        print("=> Pick Linear exponential Model since it returns smallest RMSE" + "\n")
        linear_exponential(data, period, prediction_period, confidence, best=1)
    elif min(Multi_rmse, Add_rmse, Linear_rmse) == Add_rmse:
        print("=> Pick Optimized Holt-Winter's Additive Model since it returns smallest RMSE" + "\n")
        additive(data, period, prediction_period, confidence, best=1)
    elif min(Multi_rmse, Add_rmse, Linear_rmse) == Multi_rmse:
        print("=> Pick Optimized Holt-Winter's Multiplicative Model"
              + "since it returns smallest RMSE" + "\n")
        multiplicative(data, period, prediction_period, confidence, best=1)


# Call best function to run the program
best("monthly", 12, 0.5)


'''
Test cases
'''
# Sample historic data for test cases
his_dt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
his_dt1 = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
his_dt2 = [1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4]


def RMSE_test():
    # test RMSE function
    assert(RMSE([0.1, 0.1, 0.1], his_dt, "additive", 12) >= 3)
    assert(RMSE([0.1, 0.1, 0.1], his_dt, "multiplicative", 12) >= 3)
    assert(RMSE([0.1, 0.1], his_dt, "linear_exponential", 12) <= 1)


def multiplicative_test():
    # test muplicative function
    assert(multiplicative(his_dt, 12, 12, 0.5) >= 1)
    assert(multiplicative(his_dt1, 12, 12, 0.5) <= 1)
    assert(multiplicative(his_dt2, 12, 12, 0.5) <= 1)


def additive_test():
    # test additive function
    assert(additive(his_dt, 12, 12, 0.5) >= 1)
    assert(additive(his_dt1, 12, 12, 0.5) <= 1)
    assert(additive(his_dt2, 12, 12, 0.5) <= 1)


def linear_test():
    # test lineart function
    assert(linear_exponential(his_dt, 12, 12, 0.5) <= 1)
    assert(linear_exponential(his_dt1, 12, 12, 0.5) >= 1)
    assert(linear_exponential(his_dt2, 12, 12, 0.5) >= 1)


def draw_his_test():
    # test draw increment period of historic data
    his_dt_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    his_dt1_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    his_dt2_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert(draw_his(his_dt) == his_dt_test)
    assert(draw_his(his_dt1) == his_dt1_test)
    assert(draw_his(his_dt2) == his_dt2_test)


def best_test():
    # test best function even the function does not have any value return, I still want
    # to put the test in here to check if the min method is true or not
    assert(min(additive(his_dt, 12, 12, 0.5), multiplicative(his_dt, 12, 12, 0.5),
               linear_exponential(his_dt, 12, 12, 0.5)) == linear_exponential(his_dt, 12, 12, 0.5))
    assert(min(additive(his_dt1, 12, 12, 0.5), multiplicative(his_dt1, 12, 12, 0.5),
               linear_exponential(his_dt1, 12, 12, 0.5)) == multiplicative(his_dt1, 12, 12, 0.5))
    assert(min(additive(his_dt2, 12, 12, 0.5), multiplicative(his_dt2, 12, 12, 0.5),
               linear_exponential(his_dt2, 12, 12, 0.5)) == additive(his_dt2, 12, 12, 0.5))


if __name__ == "__main__":
    # run test cases
    RMSE_test()
    multiplicative_test()
    additive_test()
    linear_test()
    best_test()
    draw_his_test()
    print("Everything passed")
