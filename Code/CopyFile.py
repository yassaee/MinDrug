#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:25:59 2021

@author: fatemeh
"""

#be name khoda
from shutil import copyfile
import os
kfold=10
directory="./Data/CCLE/CrossValidations/{:d}Fold".format(kfold)
if not os.path.exists(directory):
    os.makedirs(directory)
for iters in range(50):
    directory="./Data/CCLE/CrossValidations/{:d}Fold/Iter{:d}".format(kfold,iters+1)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print ('iters:',iters)
    for cv_num in range(kfold):
        print ('########CVNum:',cv_num+1,'########')
        src="./CrossFiles_GDSC_CCLE3/Iter{:d}/cv_index_{:d}.csv".format(iters+1,cv_num+1)
        dst="./Data/CCLE/CrossValidations/{:d}Fold/Iter{:d}/cv_index_{:d}.csv".format(kfold,iters+1,cv_num+1)
        copyfile(src, dst)
        