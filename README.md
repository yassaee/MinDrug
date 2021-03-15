# MinDrug
MinDrug is a method for predicting anti-cancer drug response by finding an optimal subset of drugs that have the most similarity with other drugs. This method predicts IC50 for a new cell-line.

## Run MinDrug
MinDrug is based on python 3.6 and upper

### Required libraries:
- Argparse
- Pandas
- Numpy
- Sklearn
- Math
- Os
 
**Note:**
Please make sure that these libraries are installed.
### Run 
At the first, Run `Sub_drug.py` as follows:
#### Sub_drug
**Input files and Parameters**
- `Train_path`: contains a matrix of real values of log IC50 where rows are cell lines and columns are drugs.
- `Theta`: is a value such that two drugs are similar, if the similarty between them is bigger or equal of Theta.
- `out_dir`: demonstrates output folder such that the output files save there.

**Output files**
- `{:out_dir}/Sub_drug.csv`: This file consists of the name and index of drugs that have most similarity with other drugs and the number of them is minimal (Sub_drug).
-  `{:out_dir}/Sub_drug.pickle`: This file consists of index of drugs in Sub_drug and IC50 matrix of training data that is needed to execute following modules.

**Command**
```
python ./Code/Sub_drug.py Train_path=./Sample_Data/Train.csv Theta=0.67 out_dir=Results
```
Users have two options to run MinDrug. If the IC50 of drugs in Sub_drug are available for cell-lines in Test file in Test file, run `Main_MinDrug.py`. Else, run `Sim_MinDrug.py`.
#### Main_MinDrug
- `Test_path`: contains a matrix where rows are cell lines and columns are drugs.
- `Pickle_path`: is a pickle file that get from `Sub_drug.py`. 
- `out_dir`: demonstrates output folder such that the output files save there.

**Output files**
- `{:out_dir}/Predict.csv`: This file is the predicted IC50 values for all drugs in each cell-lines in Test file. 

**Command**
```
python ./Code/Main_MinDrug.py Test_path=./Sample_Data/Test.csv Pickle_path=./Results/Sub_drug.pickle out_dir=Results
```
#### Sim_MinDrug
- `Test_path`: contains a matrix where rows are cell lines and columns are drugs.
- `Pickle_path`: is a pickle file that get from `Sub_drug.py`. 
- - `SimCell_path`: contains the similarity matrix between cell-lines.
- `out_dir`: demonstrates output folder such that the output files save there.

**Output files**
- `{:out_dir}/Predict.csv`: This file is the predicted IC50 values for all drugs in each cell-lines in Test file. 

**Command**
```
python ./Code/Sim_MinDrug.py Test_path=./Sample_Data/Test.csv Pickle_path=./Results/Sub_drug.pickle Test_path=./Sample_Data/Sim_Cell_line.csv out_dir=Results
```
