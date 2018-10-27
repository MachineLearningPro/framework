# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def performance(y_pred,yb):
    sign_2=y_pred+yb;
    #error_b=np.where(sign_1==1)
    true_h=np.where(sign_2==2)
    true_b=np.where(sign_2==-2)
    #=len(error_b[0])/
    recall=len(true_h[0])/len(yb[yb==1])
    precision=len(true_h[0])/len(y_pred[y_pred==1])
    accuracy=(len(true_h[0])+len(true_b[0]))/len(yb)
    return recall,precision,accuracy

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient,loss=compute_gradient(y,tx,w)
        w=w-gamma*gradient
    return w,loss
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size):
    w=initial_w
    for n_iter in range(max_iters):
        gradient,loss=compute_stoch_gradient(y,tx,w,batch_size)
        w=w-gamma*gradient
    return w,loss

def least_squares(y, tx):
    w=np.linalg.inv(tx.T@tx)@tx.T@y
    e=y-tx@w
    loss=e@e/2/len(y) 
    return w,loss

def ridge_regression(y, tx, lambda_):
    w=np.linalg.inv(tx.T@tx+2*len(y)*lambda_*np.eye(tx.shape[1]))@tx.T@y;
    e=y-tx@w
    loss=e@e/2/len(y)
    return w,loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient,loss=compute_logis_gradient(y,tx,w)
        w=w-gamma*gradient
    return w,loss

def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        gradient,loss=compute_logis_gradient(y,tx,w)
        w=w-gamma*gradient
    return w,loss


def build_model_data(data,degree):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(data)
    #r_data=np.delete(data,range(5,8),1)
    r_data=data.copy()

    t_data=data
    t_data = np.c_[np.ones(num_samples), r_data]
    for i in range(len(r_data.T)):
        for j in range(2,degree+1):
            t_data=np.c_[t_data,r_data[:,i]**(j)]
    return t_data

def compute_logis_gradient(y,tx,w):
    e=y-sigma(tx@w);
    loss=e@e/(2*len(y));
    #s=tx@w
    gradient=-tx.T@e/len(y)
    
    return gradient,loss

def sigma(a):
    return 1/(1+np.exp(-a))
    
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e=y-tx@w;
    loss=e@e/(2*len(y));
    gradient=-tx.T@e/len(y);
    #e=y-tx@w;
    #loss=0;
    #for i in range(len(y)):
    #    loss=loss+abs(e[i])
    #loss=loss/len(y)/2;
    #for i in range(len(y)):
    #    if e[i]==0:
    #        ;
    #    else:
    #        e[i]=e[i]/np.abs(e[i])
    #gradient=-tx.T@e/2/len(y) 
    return gradient,loss




    
    
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w,batch_size):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    for shuffled_y,shuffled_tx in batch_iter(y,tx,batch_size):
        shuffled_tx=shuffled_tx[0]
        e=shuffled_y-shuffled_tx@w
        loss=e*e
        gradient=-2*e*shuffled_tx
        
       # e=shuffled_y-shuffled_tx@w
       # loss=abs(e)
       # if e==0:
       #     ;
      #  else:
      #      e=e/np.abs(e)
      #  gradient=-e*shuffled_tx;
    
    return gradient,loss

def clean_data():
# set file path
#FILE_PATH = r'./'
    train_name = 'train.csv'
    test_name = 'test.csv'

# read csv data
#train_data = pd.read_csv(FILE_PATH + train_name)
#test_data = pd.read_csv(FILE_PATH + test_name)
    train_data = pd.read_csv(train_name)
    test_data = pd.read_csv(test_name)

# delete -999
    unkonwn = -999
    cleaned_training = train_data.copy()

    for ct_i in range(2,len(train_data.columns)):
        checked_column = cleaned_training.columns[ct_i]
        cleaned_training.loc[cleaned_training[checked_column] == unkonwn, cleaned_training.columns[ct_i]] = np.nan

# extract s & s particles
    B_particle = cleaned_training.Prediction == 'b'
    S_particle = cleaned_training.Prediction == 's'

    cleaned_b_particle = cleaned_training[B_particle]
    cleaned_s_particle = cleaned_training[S_particle]

# means of two kinds particles
    means_b = cleaned_b_particle.mean(skipna=True)
    means_s = cleaned_s_particle.mean(skipna=True)

# data recovery
    for ct_i in range(2,len(train_data.columns)):
        checked_columns = cleaned_training.columns[ct_i]
        cleaned_b_particle.loc[ np.isnan(cleaned_b_particle[checked_columns]) , checked_columns] = means_b[checked_columns]
        
        cleaned_s_particle.loc[ np.isnan(cleaned_s_particle[checked_columns]) , checked_columns] = means_s[checked_columns]

    data_b = cleaned_b_particle.drop(['Id','Prediction'], axis=1).values
    data_h = cleaned_s_particle.drop(['Id','Prediction'], axis=1).values
    print(data_h[3])
    label=train_data['Prediction'].values
    yb = np.ones(len(label))
    yb[np.where(label=='b')] = -1
    label=yb
    data=train_data.drop(['Id','Prediction'], axis=1).values

    return data_b,data_h,yb,data

def shrink_data(label,data,train_size):
    label_train=label[0:train_size]
    data_train=data[0:train_size]
    return label_train,data_train
    
