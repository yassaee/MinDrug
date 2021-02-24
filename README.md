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
```
python ./Code/Sub_drug.py Train_path=./Sample_Data/Train.csv Theta=0.67 out_dir=Results
```

Users have two options to run MinDrug. 

