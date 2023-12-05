import pandas as pd
import numpy as np
import random

import sys 
import os
sys.path.append(os.path.dirname(__file__)+'/..')

BASE_DIR = '.'

import argparse
parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument("-i", "--InputList", type=str, required = True, help = "Input List, should contain Path, ID, Y")
parser.add_argument("-N", "--SaveName", type=str, default = "Raw", help = '''Save list name
For example: {BASE_DIR}/List/Train_list_{FoldNum}_RLPA
SaveName should contain FoldNum if FiveFold is True
''')
parser.add_argument("-O", "--OutDir", type=str, default = f"{BASE_DIR}/List", )
parser.add_argument("-B", "--RemoveBaseDir", type=int, default = 1, help = "Remove Base direct or not, defualt is True")
parser.add_argument("-F", "--FiveFold", type=int, default = 1, help = "Five Fold or not, defualt is True")
parser.add_argument("-R", "--Ratio", type=str, default = "4:1:1", help = "Ratio to split  int only  Train:Val:Test ")
parser.add_argument("-yc", "--YColumns", type=str, default = "Y", help = "Y columns ")
args = parser.parse_args()

DATA_INFO = args.InputList
Five_fold = bool(args.FiveFold)
OutDir = args.OutDir
if not os.path.exists(OutDir):
    os.mkdir(OutDir)
SaveName = args.SaveName

ratio = [ int(i) for i in args.Ratio.split(":")]

DA = pd.read_csv(DATA_INFO)
DA = DA.sample(frac=1).reset_index()

uniq_value = pd.unique(DA['ID'])
random.shuffle(uniq_value)
To_devide_idx = len(uniq_value) // sum(ratio)


if bool(args.RemoveBaseDir):
    DA['Path'] = DA['Path'].str.replace(f"{BASE_DIR}/",'')

#col_list = ['Path','ID','Y']
if Five_fold: 
    SaveNameTest = f"{OutDir}/Test_list_{SaveName}"
    To_devide_idx=int(To_devide_idx*ratio[2])
    Test_idx = uniq_value[0:To_devide_idx]
    Test = DA.loc[DA['ID'].isin(Test_idx)]
    Test.to_csv(SaveNameTest, index = False)

    DA=DA.loc[~DA['ID'].isin(Test_idx)]
    print(f"Test: Ind_num {To_devide_idx}")
    print(Test[args.YColumns].value_counts())

    for FoldNum in range(5):
        uniq_value = pd.unique(DA['ID'])
        random.shuffle(uniq_value)
        To_devide_idx = (len(uniq_value) // (sum(ratio)-ratio[2]))*ratio[1]
        i = FoldNum
        start = i*To_devide_idx
        end = min([(i+1)*To_devide_idx,len(uniq_value)-1])
        Test_idx = uniq_value[start:end]   
        
        SaveNameTrain = f"{OutDir}/Train_list_{FoldNum}_{SaveName}"
        SaveNameValidate = f"{OutDir}/Val_list_{FoldNum}_{SaveName}"

        Train = DA.loc[~DA['ID'].isin(Test_idx)]
        Val = DA.loc[DA['ID'].isin(Test_idx)]
        print(f"Fold {i}")
        print(f"Train: Ind_num {len(Train.index)}")
        print(Train[args.YColumns].value_counts())
        print(f"Val: Ind_num {len(Val.index)}")
        print(Val[args.YColumns].value_counts())
        Train.to_csv(SaveNameTrain, index = False)
        Val.to_csv(SaveNameValidate, index = False)

else:
    SaveNameTest = f"{OutDir}/Test_list_{SaveName}"
    SaveNameTrain = f"{OutDir}/Train_list_{SaveName}"
    SaveNameValidate = f"{OutDir}/Val_list_{SaveName}"

    To_devide_idx_Test = int(To_devide_idx*ratio[2])
    To_devide_idx_Val = (len(uniq_value) // (sum(ratio)-ratio[2]))*ratio[1]

    Test = uniq_value[:To_devide_idx]    
    Val = uniq_value[To_devide_idx_Test:(To_devide_idx_Val+To_devide_idx_Test)]    
    Train = uniq_value[(To_devide_idx_Val+To_devide_idx_Test):] 

    Train = DA.loc[DA['ID'].isin(Train)]
    Test = DA.loc[DA['ID'].isin(Test)]
    Validate = DA.loc[DA['ID'].isin(Val)]

    print(Train[args.YColumns].value_counts())
    print(Validate[args.YColumns].value_counts())
    print(Test[args.YColumns].value_counts())
    Train.to_csv(SaveNameTrain, index = False)
    Validate.to_csv(SaveNameValidate, index = False)
    Test.to_csv(SaveNameTest, index = False)

# print("")
# print(f"GradCam from test_list")
# sector = Test.groupby(args.YColumns)

# da_list = []
# for i in sector.groups.keys():
#     da = sector.get_group(i)
#     da = da.iloc[:4,:]
#     da_list.append(da)

# GRAD = pd.concat(da_list)
# #GRAD = GRAD.drop(columns = ['index'])
# GRAD.to_csv(f"{OutDir}/gradCam_list_{SaveName}",index = False)
