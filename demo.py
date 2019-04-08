import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    #opens files and reads --> 'r'
    #with as statement will open and then close file for us
    with open(filename, 'r') as csvfile:
        #create reader variable.  .reader is built in method
        csvFileReader = csv.reader(csvfile)
        #call next before for loop to skip first row of file (it doesn't contain data)
        next(csvFileReader)
        for row in csvFileReader:
            #removes / from dates and selects [2] for Day
            dates.append(int(row[0].split('/')[2]))
            prices.append(float(row[3]))
    return

#build predictive model and graph it
#will be n x 1 matrix
def predict_prices(dates, prices, x):
    #use numpy to format list
    dates = np.reshape(dates, (len(dates), 1))
    # dates = np.reshape(-1, 1)
    #SVR support vector regression model.  SVM that predicts a value instead of making a classification. 
    #c=1e3 ---sci notation for 1k
    svr_lin = SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel= 'poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel= 'rbf', C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    #making scatter plot
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial Model')
    #setting axis labels and legend
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()

    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('HistoricalQuotes.csv')

predicted_price = predict_prices(dates, prices, 22)

print(predicted_price)
