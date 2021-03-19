#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In The Name of God
@author: fatemeh
"""
import argparse
import pandas as pd
import numpy as np
import math
import os
import pickle
from numpy import linalg as LA
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
def run(Train_dir,Theta,out_sub,out_pickle):
    df_tr = pd.read_csv(Train_dir,index_col =0)
    drug=df_tr.columns
    cell_train=list(df_tr.index)
    IC50 =np.matrix(df_tr)
    IC50_T1=np.transpose(IC50)
    select_index,drug1=Sub_drug(IC50_T1, drug,Theta)
    outfile_pickle = open(out_pickle,'wb')
    pickle.dump(IC50,outfile_pickle)
    pickle.dump(cell_train,outfile_pickle)
    pickle.dump(drug,outfile_pickle)
    pickle.dump(select_index, outfile_pickle)
    outfile_pickle.close()
    columns_sub=['Index']
    sub_DataFrame=pd.DataFrame(select_index,index=drug1,columns=columns_sub)
    sub_DataFrame.to_csv(out_sub)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("Train_path_file")
    parser.add_argument("Theta")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    Theta=float(args.Theta.split('=')[1])
    Train_dir=args.Train_path_file.split('=')[1]
    out_dir=args.out_dir.split('=')[1]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_sub="{:s}/Sub_drug.csv".format(out_dir)
    out_pickle="{:s}/Sub_drug.pickle".format(out_dir)
    run(Train_dir,Theta,out_sub,out_pickle)
#######
main()
