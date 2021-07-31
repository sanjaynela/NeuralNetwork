import pickle
import gzip
import numpy as np
import sys
import csv
import pandas as pd
import time

def load_data_prompts():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]

    tr_d=pd.read_csv(file1, sep=',',header=None)
    tr_img = tr_d.values
    # print(tr_d)
    # print(np.shape(tr_d)) #(60000,784)
    va_d=pd.read_csv(file2, sep=',',header=None)
    tr_label = va_d.values
    # print(va_d)
    # print(np.shape(va_d)) #(60000,1)
    te_d=pd.read_csv(file3, sep=',',header=None)
    test_img = te_d.values
    # print(te_d)
    # print(np.shape(te_d)) #(10000,784)
    return (tr_img, tr_label, test_img)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
# file4 = "test_label.csv"

tr_d=pd.read_csv(file1, sep=',',header=None)
tr_img = tr_d.values
# print(tr_d)
# print(np.shape(tr_d)) #(60000,784)
tr_l=pd.read_csv(file2, sep=',',header=None)
tr_label = tr_l.values
# print(va_d)
# print(np.shape(va_d)) #(60000,1)
te_img=pd.read_csv(file3, sep=',',header=None)
test_img = te_img.values

# te_label = pd.read_csv(file4, sep=',',header=None)
# test_label = te_label.values

def activation_function(x):
    result = 1 / (1 + np.exp(-x))
    return result

def activation_function_derivative(x):
    activation_result = 1 / (1 + np.exp(-x))
    result = activation_result * (1-activation_result)
    return result

def execute_multiple_class_loss(Y, Y_output):
    Loss_sum = np.sum(np.multiply(Y, np.log(Y_output)))
    num_records = Y.shape[1]
    loss = -(1/num_records) * Loss_sum
    return loss

# start_time = time.time()

X = tr_img / 255 #Normalize input image date (Pixels are in range from 0 to 255)
X1 = test_img / 255

y = tr_label
num_labels = 10
#Number of rows in training label (60000)
num_cols = X.shape[1]

#Training Data

#Number of rows in training data
num_of_records = tr_d[0].size
# print(num_of_records) #60000
#Execute transposition for training and testing data
X_training = X.T
X_testing = X1.T
#Make hot-code representation of y_train
targets = y.reshape(-1)
Y_training = np.eye(num_labels)[targets].T

# print(Y_training) #(10,60000)
#Retrieve num_rows which is actually no ofnpcols in training data
num_rows = X_training.shape[0]
#Initialize numb of hidden layers and learning rate
lr = 1
num_hidden_layers = 80
#Initialize weights and bias parameters
first_weight = np.random.randn(num_hidden_layers, num_rows) / np.sqrt(num_hidden_layers)
first_bias = np.zeros((num_hidden_layers, 1))
# weight_middle = np.random.randn(num_rows,num_rows) / np.sqrt(num_rows)
# bias_middle = np.zeros((num_hidden_layers, 1))
second_weight = np.random.randn(num_labels, num_hidden_layers) / np.sqrt(num_labels)
second_bias = np.zeros((num_labels, 1))

for epoch in range(2200):
    #Forward-propagation - Update input and activiation layer parameters
    initial_product = np.dot(first_weight,X_training)
    input_first = initial_product + first_bias
    first_activation = activation_function(input_first)
    second_input = np.dot(second_weight,first_activation) + second_bias

    exponential_of_last_layer = np.exp(second_input)
    # print(np.shape(exponential_of_last_layer))
    sum_of_normalizations = np.sum(np.exp(second_input), axis=0)
    final_activation = exponential_of_last_layer/ sum_of_normalizations

    #Calculate loss at output after forward-propagation is completed. 
    cost = execute_multiple_class_loss(Y_training, final_activation)

    #Back-propagation
    #Update gradients of 2nd input layer and 2nd weight, bias parameters
    #Calculate derivation of entropy loss in output layer
    entropy_loss_derivative = final_activation-Y_training
    # print(np.shape(entropy_loss_derivative))
    product_rule2 = np.dot(entropy_loss_derivative, first_activation.T)
    loss_gradient_weight_second = product_rule2/num_of_records
    sum_rule2 = np.sum(entropy_loss_derivative, axis=1, keepdims=True)
    loss_gradient_bias_second = sum_rule2/num_of_records

    #Update gradients of 1st activation layer, 1st input layer and 1st weight, bias parameters
    delta_first_activation = np.dot(second_weight.T, entropy_loss_derivative)
    product_rule3 = activation_function_derivative(input_first)
    delta_input_first = delta_first_activation * product_rule3
    product_rule4 = np.dot(delta_input_first, X_training.T)
    loss_gradient_weight_first = product_rule4/num_of_records
    sum_rule3 = np.sum(delta_input_first, axis=1, keepdims=True)
    loss_gradient_bias_first = sum_rule3/num_of_records
    
    #Update initial bias and weight parameters using computed gradient
    #1st layer
    first_delta_bias = lr * loss_gradient_bias_first
    first_delta_weight = lr * loss_gradient_weight_first
    first_bias = first_bias - first_delta_bias
    first_weight = first_weight - first_delta_weight
    #2nd layer
    second_delta_bias = lr * loss_gradient_bias_second
    second_delta_weight= lr * loss_gradient_weight_second
    second_bias = second_bias - second_delta_bias
    second_weight = second_weight - second_delta_weight
    #DELETE BELOW LINE - FOR DEBUGGING
    if (epoch % 100 == 0):
        print("Epoch", epoch, "cost: ", cost)

#Compute result for 2nd activation layer using updated weight and bias parameters
#Result for 2nd activation layer is the predicted result
#Y = Wx + B
first_result = np.dot(first_weight, X_testing)
input_first = first_result + first_bias
first_activation = activation_function(input_first)
second_result = np.dot(second_weight, first_activation)
second_input = second_result + second_bias
exponential_of_last_layer = np.exp(second_input)
sum_of_normalizations = np.sum(np.exp(second_input), axis=0)
final_activation = exponential_of_last_layer/ sum_of_normalizations

#Use argmax to convert one hot encoding back to categorical
final_results = np.argmax(final_activation, axis=0)
# print(final_results)
final_results = final_results.astype(int)
np.savetxt("test_predictions.csv", final_results , delimiter="," , fmt="%d")
# end_time = time.time()
# print(end_time - start_time)