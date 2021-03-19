"""
#In The Name of God

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
def find_sim_cell(row_cell,cell_train):
    print('celltr',cell_train)
    max_sim_all=sorted(cell_train, key=lambda k: row_cell[k],reverse=True)
    print('&&&&&&&&&&&&&&&&&&&&',max_sim_all)
    return max_sim_all[:15]
def Train_Predict(IC50,drug,select_index,Cell_Sim,cell,cell_train):
    P_IC50=[[0.0 for i in range(len(drug))]for j in range(len(cell))]
    alphas=0.01
    x_train=[]
    for ind_x in range(len(IC50)):
        aux=[]
        for ind_j in range(len(drug)):
            if ind_j in select_index:
                aux.append(IC50[ind_x,ind_j])
        x_train.append(aux)
    x_test=[]
    for ind_x in range(len(cell)):
        aux=[]
        Sim_cell=find_sim_cell(Cell_Sim[cell[ind_x]],cell_train)
        for ind_j in range(len(drug)):
            if ind_j in select_index:
                i_d=[]
                s_sim=0.0
                for j_sim in range(len(Sim_cell)):
                    i_d.append(Cell_Sim[cell[ind_x]][Sim_cell[j_sim]]*IC50[cell_train.index(Sim_cell[j_sim]),ind_j])
                    s_sim+=Cell_Sim[cell[ind_x]][Sim_cell[j_sim]]
                aux.append(float(float(sum(i_d))/float(s_sim)))
        x_test.append(aux)
    elastic=ElasticNet(alpha=alphas).fit(x_train, IC50)
    ypred = elastic.predict(x_test)
    for lx in range(len(ypred)):
        P_IC50[lx]=ypred[lx]
    return P_IC50
def run(Test_dir,cell_dir,out_file,pickle_file):
    infile = open(pickle_file,'rb')
    df_tr=pickle.load(infile)
    cell_train=pickle.load(infile)
    drug=pickle.load(infile)
    select_index=pickle.load(infile)
    IC50 =np.matrix(df_tr)
    df_ts = pd.read_csv(Test_dir,index_col =0)
    cell_test=df_ts.index
    test =np.matrix(df_ts)
    df_cell = pd.read_csv(cell_dir,index_col =0)
    Cell_Sim=df_cell.to_dict('dict')
    P_IC50=Train_Predict(IC50,drug,select_index,Cell_Sim,cell_test,cell_train)
    Pred_IC50=np.matrix(P_IC50)
    predict_DataFrame=pd.DataFrame(Pred_IC50,index=cell_test,columns=drug)
    predict_DataFrame.to_csv(out_file)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Test_path_file")#TODO:********
    parser.add_argument("Pickle_path")
    parser.add_argument("Cell_line_Similarity_path_file")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    Test_dir=args.Test_path_file.split('=')[1]
    pickle_file=args.Pickle_path.split('=')[1]
    cell_dir=args.Cell_line_Similarity_path_file.split('=')[1]
    out_dir=args.out_dir.split('=')[1]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file="{:s}/Predict_Sim.csv".format(out_dir)
    run(Test_dir,cell_dir,out_file,pickle_file)
#######
main()
