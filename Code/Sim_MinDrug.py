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
def find_sim_cell(row_cell,cell_train):
    max_sim_all=sorted(cell_train, key=lambda k: row_cell[k],reverse=True)
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
def Sub_drug(IC50_T1,drug,Theta):
    Sim_IC50=[[0 for i in range(len(drug))]for j in range(len(drug))]
    Aux_IC50=[[0 for i in range(len(drug))]for j in range(len(drug))]
    for i in range(len(drug)):
        a=IC50_T1[i]
        for j in range(len(drug)):
            b=IC50_T1[j]
            nr=LA.norm(np.array(b)-np.array(a))
            Aux_IC50[i][j]=nr
    MAX=math.ceil(max(max(Aux_IC50)))
    for i in range(len(drug)):
        Sim_IC50[i]=[1.0-float(float(x_aux-min(min(Aux_IC50)))/float(MAX-min(min(Aux_IC50)))) for x_aux in Aux_IC50[i]]
    cal_SimIC50=[0 for i in range(len(Sim_IC50))]
    selected=[]
    select_index=[]
    counts=0
    while cal_SimIC50!=[-1 for i in range(len(Sim_IC50))]:
        max_norm=-1
        max_i=-1
        max_vec=[0 for i in range(len(Sim_IC50))]
        for i in range(len(Sim_IC50)):
            if cal_SimIC50[i]==0:
                x2=[]
                for ji in range(len(Sim_IC50)):
                    if cal_SimIC50[ji]==0 and Sim_IC50[i][ji]>=0.5 and i!=ji:
                        x2.append(Sim_IC50[i][ji])
                x1=np.array(x2)
                if max_norm<len(x1):
                    max_norm=len(x1)
                    max_i=i
                    max_vec=Sim_IC50[i]
        if max_i != -1:
            selected.append(max_vec)
            select_index.append(max_i)
            cal_SimIC50[max_i]=-1
            counts += 1
        for i in range(len(Sim_IC50)):
            if cal_SimIC50[i]==0 and max_i!= -1:
                if Sim_IC50[max_i][i]>=Theta:
                    cal_SimIC50[i]=-1
                    counts+=1
    drug1=[drug[i] for i in select_index]
    print( '***',drug1,'***')
    return select_index,drug1
def run(Train_dir,Test_dir,cell_dir,Theta,out_file,out_sub):
    df_tr = pd.read_csv(Train_dir,index_col =0)
    drug=df_tr.columns
    cell_train=list(df_tr.index)
    IC50 =np.matrix(df_tr)
    IC50_T1=np.transpose(IC50)
    df_ts = pd.read_csv(Test_dir,index_col =0)
    cell=df_ts.index
    test =np.matrix(df_ts)
    df_cell = pd.read_csv(cell_dir,index_col =0)
    Cell_Sim=df_cell.to_dict('dict')
    select_index,drug1=Sub_drug(IC50_T1, drug,Theta)
    P_IC50=Train_Predict(IC50,drug,select_index,Cell_Sim,cell,cell_train)
    Pred_IC50=np.matrix(P_IC50)
    predict_DataFrame=pd.DataFrame(Pred_IC50,index=cell,columns=drug)
    predict_DataFrame.to_csv(out_file)
    columns_sub=['Index']
    sub_DataFrame=pd.DataFrame(select_index,index=drug1,columns=columns_sub)
    sub_DataFrame.to_csv(out_sub) 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Train_path_file")
    parser.add_argument("Test_path_file")#TODO:********
    parser.add_argument("Cell_line_Similarity_path_file")
    parser.add_argument("Theta")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    Theta=float(args.Theta.split('=')[1])
    Train_dir=args.Train_path_file.split('=')[1]
    Test_dir=args.Test_path_file.split('=')[1]
    cell_dir=args.Cell_line_Similarity_path_file.split('=')[1]
    out_dir=args.out_dir.split('=')[1]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file="{:s}/Predict.csv".format(out_dir)
    out_sub="{:s}/Sub_drug.csv".format(out_dir)
    run(Train_dir,Test_dir,cell_dir,Theta,out_file,out_sub)
#######
main()