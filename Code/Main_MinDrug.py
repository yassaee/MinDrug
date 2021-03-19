#In The Name of God
"""
Created on Mon Dec 28 13:14:37 2020

@author: fatemeh
"""
import argparse
import pandas as pd
import numpy as np
import math
import os
from sklearn.linear_model import ElasticNet
from numpy import linalg as LA
import pickle
def Train_Predict(IC50,test,drug,select_index):
    P_IC50=[[0.0 for i in range(len(drug))]for j in range(len(test))]
    alphas=0.01
    x_train=[]
    for ind_x in range(len(IC50)):
        aux=[]
        for ind_j in range(len(drug)):
            if ind_j in select_index:
                aux.append(IC50[ind_x,ind_j])
        x_train.append(aux)
    x_test=[]
    for ind_x in range(len(test)):
        aux=[]
        for ind_j in range(len(drug)):
            if ind_j in select_index:
                aux.append(test[ind_x,ind_j])
        x_test.append(aux)
    elastic=ElasticNet(alpha=alphas).fit(x_train, IC50)
    ypred = elastic.predict(x_test)
    for lx in range(len(ypred)):
        P_IC50[lx]=ypred[lx]
    return P_IC50
def run(Test_dir,out_file,pickle_file):
    infile = open(pickle_file,'rb')
    df_tr=pickle.load(infile)
    cell_train=pickle.load(infile)
    drug=pickle.load(infile)
    select_index=pickle.load(infile)
    IC50 =np.matrix(df_tr)
    IC50_T1=np.transpose(IC50)
    df_ts = pd.read_csv(Test_dir,index_col =0)
    cell=df_ts.index
    test =np.matrix(df_ts)
    P_IC50=Train_Predict(IC50,test,drug,select_index)
    Pred_IC50=np.matrix(P_IC50)
    predict_DataFrame=pd.DataFrame(Pred_IC50,index=cell,columns=drug)
    predict_DataFrame.to_csv(out_file)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Test_path_file")#TODO:*******
    parser.add_argument("Pickle_path")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    Test_dir=args.Test_path_file.split('=')[1]
    pickle_file=args.Pickle_path.split('=')[1]
    out_dir=args.out_dir.split('=')[1]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file="{:s}/Predict_Main.csv".format(out_dir)
    run(Test_dir,out_file,pickle_file)
#######
main()
