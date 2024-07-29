import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
import numpy as np
from sklearn.metrics import auc


#rc('text', usetex=True)
#plt.rc('text', usetex=True)

def Recall(TP, FN):
    try:
        return TP/(TP+FN)
    except ZeroDivisionError:
        return 0

def Precision(TP, FP):
    try:
        return TP/(TP+FP)
    except ZeroDivisionError:
        return 0

def GMean(TP, FN, TN, FP):
    try:
        return math.sqrt((TP*TN)/((TP+FN)*(FP*TN)))
    except ZeroDivisionError:
        return 0

def F1Score(TP, FN, FP):
    R = Recall(TP, FN)
    P = Precision(TP, FP)
    try:
        return (2*R*P)/(P+R)
    except ZeroDivisionError:
        return 0

def FBScore(TP, FN, FP, B):
    R = Recall(TP, FN)
    P = Precision(TP, FP)
    try:
        return ((1+B**2)*R*P)/((B**2)*P+R)
    except ZeroDivisionError:
        return 0

def PLR(TP, TN, FN, FP):
    try:
        return (TP/(TP+FN))/(FP/(FP+TN))
    except ZeroDivisionError:
        return 0

def TPR(TP, FN):
    try:
        return TP/(TP+FN)
    except ZeroDivisionError:
        return 0

def FPR(FP, TN):
    try:
        return FP/(FP+TN)
    except ZeroDivisionError:
        return 0

def compute_all(TP, TN, FP, FN):
    R = Recall(TP, FN)
    P = Precision(TP, FP)
    G = GMean(TP, FN, TN, FP)
    FS = F1Score(TP, FN, FP)
    FS_2 = FBScore(TP, FN, FP, 2)
    FS_0_5 = FBScore(TP, FN, FP, 0.5)
    LR = PLR(TP, TN, FN, FP)
    TruePositive = TPR(TP, FN)
    FalsePositive = FPR(FP, TN)
    return R, P, G, FS, FS_2, FS_0_5, LR, TruePositive, FalsePositive

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

path = "XGBoostResult/"
files = os.listdir(path)

AreaBelowCurves = {}

try:
    file_table = open("LogisticRegressionTable.tex", "w")
except:
    sys.exit()
    


file_table.write(r'\begin{table}' + "\n")
#fMainArticle.write(r'\resizebox{2\columnwidth}{!}{' + "\n")
file_table.write(r'\caption{XGBoost}\label{tab:XGBoost}\centering' + "\n")
file_table.write(r'\begin{tabular}{?c|c?c|c?c|c?}' + "\n")
file_table.write(r'\noalign{\hrule height 1.5pt}'+ "\n")
file_table.write(r'\multirow{2}{*}{\textbf{Dim}} & \multirow{2}{*}{\textbf{Feat}} & \multicolumn{2}{c?}{Training} & \multicolumn{2}{c?}{Test} \\' + "\n" )
file_table.write(r'\cline{3-6}' + "\n" )
file_table.write(r' &  & \textbf{ROC} & \textbf{ROC PR} & \textbf{ROC} & \textbf{ROC PR} \\' + "\n" )
file_table.write(r'\noalign{\hrule height 1.5pt}'+ "\n")
   

DatasetBestThreshold = {}
DatasetBestF2 = {}


DatasetBestThresholdTrain = {}
DatasetBestF2Train = {}

for file_name in files:
    field =  file_name[:-4].split("_")
    OutputPath = "./" + field[2] + "/" + field[3]
    Path(OutputPath).mkdir(parents=True, exist_ok=True)

    Res = pd.read_csv(path +file_name, index_col=0).to_dict()
    ListOfConsideredMetrics = ["Recall", "Precision", "GMean", "F1Score", "F2Score", "F05Score", "LR+", "TPR", "FPR"]
    MetricsTraining = {}
    MetricsTest = {}
    for el in ListOfConsideredMetrics:
        MetricsTraining[el] = {}
        MetricsTest[el] = {}

    if not field[2] in DatasetBestThreshold.keys():
        DatasetBestThreshold[field[2]] = {}
        DatasetBestF2[field[2]] = {}
    BestTh = 0
    BestF2Score = 0
    
    if not field[2] in DatasetBestThresholdTrain.keys():
        DatasetBestThresholdTrain[field[2]] = {}
        DatasetBestF2Train[field[2]] = {}
    BestThTrain = 0
    BestF2ScoreTrain = 0
    
    for i in range(len(Res['threashold'])):
        Thr = Res['threashold'][i]
        TP_Train = Res['tpTrain'][i]
        FP_Train = Res['fpTrain'][i]
        FN_Train = Res['fnTrain'][i]
        TN_Train = Res['tnTrain'][i]
        TP_Test = Res['tpTest'][i]
        FP_Test = Res['fpTest'][i]
        FN_Test = Res['fnTest'][i]
        TN_Test = Res['tnTest'][i]

        M_Train = compute_all(TP_Train, TN_Train, FP_Train, FN_Train)
        M_Test = compute_all(TP_Test, TN_Test, FP_Test, FN_Test)
        for j in range(len(ListOfConsideredMetrics)):
            MetricsTraining[ListOfConsideredMetrics[j]][Thr] = M_Train[j]
            MetricsTest[ListOfConsideredMetrics[j]][Thr] = M_Test[j]
        if i == 0 or M_Test[4] > BestF2Score:
            BestTh = Thr
            BestF2Score = M_Test[4]    
        if i == 0 or M_Train[4] > BestF2ScoreTrain:
            BestThTrain = Thr
            BestF2ScoreTrain = M_Train[4]   
            
    DatasetBestThreshold[field[2]][field[3]] = BestTh   
    DatasetBestF2[field[2]][field[3]] = BestF2Score   
    DatasetBestThresholdTrain[field[2]][field[3]] = BestThTrain   
    DatasetBestF2Train[field[2]][field[3]] = BestF2ScoreTrain   
    
    DataFramePandas = pd.DataFrame.from_dict(MetricsTraining, orient='index')
    DataFramePandas.to_csv(OutputPath + "/" + file_name[:-4] + "_Training.csv")
    DataFramePandas = pd.DataFrame.from_dict(MetricsTest, orient='index')
    DataFramePandas.to_csv(OutputPath + "/" + file_name[:-4] + "_Test.csv")
    
    AreaBelowCurves[file_name[:-4]] = {}
    
    roc_score_training = auc(list(MetricsTraining["FPR"].values()), list(MetricsTraining["TPR"].values()))
    AreaBelowCurves[file_name[:-4]]["roc_score_training"] = roc_score_training
    
    roc_score_test = auc(list(MetricsTest["FPR"].values()), list(MetricsTest["TPR"].values()))
    AreaBelowCurves[file_name[:-4]]["roc_score_test"] = roc_score_test

    roc_pr_score_training = auc(list(MetricsTraining["Recall"].values()), list(MetricsTraining["Precision"].values()))
    AreaBelowCurves[file_name[:-4]]["roc_pr_score_training"] = roc_pr_score_training
    
    roc_pr_score_test = auc(list(MetricsTest["Recall"].values()), list(MetricsTest["Precision"].values()))
    AreaBelowCurves[file_name[:-4]]["roc_pr__score_test"] = roc_pr_score_test
    
    line = field[2] + r'&' + field[3] + r'&' +"{:.3f}".format(roc_score_training) + r'&' +"{:.3f}".format(roc_pr_score_training)  + r'&' +"{:.3f}".format(roc_score_test) + r'&' +"{:.3f}".format(roc_pr_score_test)+ r'\\' + "\n" + r' \hline' + " \n"
    file_table.write(line)       
  
    '''listaCol = [ "firebrick", "coral", "gold", "yellowgreen", "lightgreen", "forestgreen", "turquoise", "deepskyblue", "royalblue", "slateblue", "purple", "orchid", "hotpink"]

    for k in range(len(ListOfConsideredMetrics)):
        plt.plot(list(MetricsTraining[ListOfConsideredMetrics[k]].keys()),list(MetricsTraining[ListOfConsideredMetrics[k]].values()), color=listaCol[k], linewidth=2,label=ListOfConsideredMetrics[k])
    plt.xlabel("Thresholds", fontsize=20)
    plt.ylabel("Metrics", fontsize=20)
    plt.xscale('log')
    leg = plt.legend(loc='upper right', frameon=True, fontsize=10)
    leg.get_frame().set_facecolor('white')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_Training" + ".eps", format='eps',bbox_inches='tight')
    plt.savefig(OutputPath + "/" +file_name[:-4] + "_Training" + ".png", format='png', bbox_inches='tight')
    plt.savefig(OutputPath + "/" +file_name[:-4] + "_Training" + ".pdf", format='pdf', bbox_inches='tight')
    plt.close()

    for k in range(len(ListOfConsideredMetrics)):
        plt.plot(list(MetricsTraining[ListOfConsideredMetrics[k]].keys()),list(MetricsTraining[ListOfConsideredMetrics[k]].values()), color=listaCol[k], linewidth=2)
        plt.xlabel("Thresholds", fontsize=20)
        plt.ylabel(ListOfConsideredMetrics[k], fontsize=20)
        plt.xscale('log')
        #plt.savefig(OutputPath + "/" +file_name[:-4] + "_" + ListOfConsideredMetrics[k] + "_Training" + ".eps", format='eps',bbox_inches='tight')
        #plt.savefig(OutputPath + "/" +file_name[:-4] + "_" + ListOfConsideredMetrics[k] + "_Training" + ".png", format='png', bbox_inches='tight')
        plt.savefig(OutputPath + "/" +file_name[:-4] + "_" + ListOfConsideredMetrics[k] + "_Training" + ".pdf", format='pdf', bbox_inches='tight')
        plt.close()

    for k in range(len(ListOfConsideredMetrics)):
        plt.plot(list(MetricsTest[ListOfConsideredMetrics[k]].keys()),list(MetricsTest[ListOfConsideredMetrics[k]].values()), color=listaCol[k], linewidth=2,label=ListOfConsideredMetrics[k])
    plt.xlabel("Thresholds", fontsize=20)
    plt.ylabel("Metrics", fontsize=20)
    leg.get_frame().set_facecolor('white')
    plt.xscale('log')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(OutputPath + "/" +file_name[:-4] + "_Test" + ".eps", format='eps',bbox_inches='tight')
    plt.savefig(OutputPath + "/" +file_name[:-4] + "_Test" + ".png", format='png', bbox_inches='tight')
    plt.savefig(OutputPath + "/" +file_name[:-4] + "_Test" + ".pdf", format='pdf', bbox_inches='tight')
    plt.close()

    for k in range(len(ListOfConsideredMetrics)):
        plt.plot(list(MetricsTest[ListOfConsideredMetrics[k]].keys()),list(MetricsTest[ListOfConsideredMetrics[k]].values()), color=listaCol[k], linewidth=2)
        plt.xlabel("Thresholds", fontsize=20)
        plt.xscale('log')
        plt.ylabel( ListOfConsideredMetrics[k], fontsize=20)
        #plt.savefig(OutputPath + "/" +file_name[:-4] + "_" + ListOfConsideredMetrics[k] + "_Test" + ".eps", format='eps',bbox_inches='tight')
        #plt.savefig(OutputPath + "/" +file_name[:-4] + "_" + ListOfConsideredMetrics[k] + "_Test" + ".png", format='png', bbox_inches='tight')
        plt.savefig(OutputPath + "/" +file_name[:-4] + "_" + ListOfConsideredMetrics[k] + "_Test" + ".pdf", format='pdf', bbox_inches='tight')
        plt.close()



    plt.plot(list(MetricsTraining["Recall"].values()), list(MetricsTraining["Precision"].values()), color="firebrick",linewidth=2)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_RecallPrecisionTraining" + ".eps", format='eps',bbox_inches='tight')
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_RecallPrecisionTraining" + ".png", format='png',bbox_inches='tight')
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_RecallPrecisionTraining" + ".pdf", format='pdf',bbox_inches='tight')
    plt.close()

    plt.plot(list(MetricsTraining["FPR"].values()), list(MetricsTraining["TPR"].values()), color="firebrick",linewidth=2)
    plt.ylabel("True Positive Ratio", fontsize=20)
    plt.xlabel("False Positive Ratio", fontsize=20)
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_TrueFalsePositiveRatioTraining" + ".eps", format='eps',bbox_inches='tight')
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_TrueFalsePositiveRatioTraining" + ".png", format='png',bbox_inches='tight')
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_TrueFalsePositiveRatioTraining" + ".pdf", format='pdf',bbox_inches='tight')
    plt.close()

    plt.plot(list(MetricsTest["Recall"].values()), list(MetricsTest["Precision"].values()), color="firebrick",linewidth=2)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_RecallPrecisionTest" + ".eps", format='eps', bbox_inches='tight')
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_RecallPrecisionTest" + ".png", format='png', bbox_inches='tight')
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_RecallPrecisionTest" + ".pdf", format='pdf', bbox_inches='tight')
    plt.close()

    plt.plot(list(MetricsTest["FPR"].values()), list(MetricsTest["TPR"].values()), color="firebrick", linewidth=2)
    plt.ylabel("True Positive Ratio", fontsize=20)
    plt.xlabel("False Positive Ratio", fontsize=20)
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_TrueFalsePositiveRatioTest" + ".eps", format='eps',bbox_inches='tight')
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_TrueFalsePositiveRatioTest" + ".png", format='png',bbox_inches='tight')
    plt.savefig(OutputPath + "/" + file_name[:-4] + "_TrueFalsePositiveRatioTest" + ".pdf", format='pdf',bbox_inches='tight')
    plt.close()'''

DataFramePandas = pd.DataFrame.from_dict(AreaBelowCurves, orient='index')
DataFramePandas.to_csv("XGBoost.csv")

DataFramePandas = pd.DataFrame.from_dict(DatasetBestThreshold, orient='index')
DataFramePandas.to_csv("BestThresholdXGBoost.csv")
DataFramePandas.to_excel("BestThresholdXGBoost.xlsx")
DataFramePandas = pd.DataFrame.from_dict(DatasetBestF2, orient='index')
DataFramePandas.to_csv("BestF2XGBoost.csv")
DataFramePandas.to_excel("BestF2XGBoost.xlsx")

DataFramePandas = pd.DataFrame.from_dict(DatasetBestThresholdTrain, orient='index')
DataFramePandas.to_csv("BestThresholdXGBoostTrain.csv")
DataFramePandas.to_excel("BestThresholdXGBoostTrain.xlsx")
DataFramePandas = pd.DataFrame.from_dict(DatasetBestF2Train, orient='index')
DataFramePandas.to_csv("BestF2XGBoostTrain.csv")
DataFramePandas.to_excel("BestF2XGBoostTrain.xlsx")

file_table.write(r'\end{tabular}' + "\n") 
file_table.write(r'\end{table}' + "\n")
file_table.close()
file_table.close()





