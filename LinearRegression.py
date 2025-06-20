import pandas as pd
import numpy as np

file_name = str(input("Enter file to feed for training: "))

url = open(file_name)
data = pd.read_csv(url)

# Drop the missing values
data = data.dropna()

input_data = str(input("Enter input column: "))
output_data = str(input("Enter output column: "))

max_range = int(input("enter max range to view dataset: "))

# training dataset and labels
train_input = np.array(data[input_data][0:max_range]).reshape(max_range, 1)
train_output = np.array(data[output_data][0:max_range]).reshape(max_range, 1)

class LinearRegression:

    def __init__(self):
        self.parameters = {}

    def forward_propagation(self, input_data):
        m = self.parameters["m"]
        c = self.parameters["c"]

        predictions = np.multiply(m, input_data) + c
        return predictions
    
    def cost_function(self, output_data, predictions):

        cost = np.mean((output_data - predictions) ** 2)
        return cost
    
    def backward_propagation(self,input_data, output_data, predictions):

        derivatives = {}

        df = predictions - output_data
        dc = 2 * (np.mean(df))
        dm = 2 * np.mean(np.multiply(df, input_data))

        derivatives["dc"] = dc
        derivatives["dm"] = dm
        return derivatives

    def update(self, learning_rate, dc, dm):

        self.parameters["m"]  = self.parameters["m"] - (learning_rate * dm)
        self.parameters["c"]  = self.parameters["c"] - (learning_rate * dc)

    def train(self, learning_rate, input_data, output_data, iterations):

        self.parameters['m'] = np.random.uniform(0, 1) * -1
        self.parameters['c'] = np.random.uniform(0, 1) * -1

        i = 0
        while i != iterations:
            predictions = self.forward_propagation(input_data)
            cost = self.cost_function(output_data, predictions)
            print("Training Model for this dataset")
            print("cost: ", cost)

            backward_prop_data = self.backward_propagation(input_data,output_data,predictions)

            dc = backward_prop_data["dc"]
            dm = backward_prop_data["dm"]

            self.update(learning_rate, dc, dm)

            i = i + 1

        return self.parameters["m"], self.parameters["c"]

iterations = int(input("Enter number of iterations: "))
Linear_reg = LinearRegression()

m,c = Linear_reg.train(0.0001, train_input, train_output,iterations)

i = 0

while i != 10:
    test = float(input("Enter input for prediction: "))
    print((m * test) + c)