import pandas
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# LINEAR REGRESSION
X_train = []
y_train = []

X_test = 0

model = LinearRegression()
model.fit(X_train, y_train)

# Then use this for prediction
y_pred = model.predict(X_test)

# MULTIPLE LINEAR REGRESSION
df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2)