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

path = "NeuralNetwork/"
files = os.listdir(path)

AreaBelowCurves = {}

try:
    file_table = open("LogisticRegressionTable.tex", "w")
except:
    sys.exit()
    


file_table.write(r'\begin{table}' + "\n")
#fMainArticle.write(r'\resizebox{2\columnwidth}{!}{' + "\n")
file_table.write(r'\caption{NeuralNetwork}\label{tab:NeuralNetwork}\centering' + "\n")
file_table.write(r'\begin{tabular}{?c|c|c?c|c?c|c?}' + "\n")
file_table.write(r'\noalign{\hrule height 1.5pt}'+ "\n")
file_table.write(r'\multirow{2}{*}{\textbf{Dim}} & \multirow{2}{*}{\textbf{Feat}} & \multirow{2}{*}{\textbf{Layer}} & \multicolumn{2}{c?}{Training} & \multicolumn{2}{c?}{Test} \\' + "\n" )
file_table.write(r'\cline{4-7}' + "\n" )
file_table.write(r' &  & & \textbf{ROC} & \textbf{ROC PR} & \textbf{ROC} & \textbf{ROC PR} \\' + "\n" )
file_table.write(r'\noalign{\hrule height 1.5pt}'+ "\n")
   
Dimensions = [1000, 2000, 4000, 6000, 8000, 10000, 20000]
features = [2, 4, 6, 8]

ROC = {}
PR = {}
ROC_test = {}
PR_test = {}
DatasetBestThreshold = {}
DatasetBestF2 = {}
DatasetBestModel = {}
Test_metrics = {}
DatasetBestThresholdTrain = {}
DatasetBestF2Train = {}
DatasetBestModelTrain = {}
Train_metrics = {}

for dim in Dimensions:
    ROC[dim] = {}
    PR[dim] = {}
    ROC_test[dim] = {}
    PR_test[dim] = {}
    DatasetBestThreshold[dim] = {}
    DatasetBestF2[dim] = {}
    DatasetBestModel[dim] = {}
    Test_metrics[dim] = {}
    DatasetBestThresholdTrain[dim] = {}
    DatasetBestF2Train[dim] = {}
    DatasetBestModelTrain[dim] = {}
    Train_metrics[dim] = {}
    for feature in features:
        ROC[dim][feature] = {}
        PR[dim][feature] = {}
        ROC_test[dim][feature] = {}
        PR_test[dim][feature] = {}
        DatasetBestThreshold[dim][feature] = {}
        DatasetBestF2[dim][feature] = {}
        DatasetBestModel[dim][feature] = {}
        Test_metrics[dim][feature] = {}
        DatasetBestThresholdTrain[dim][feature] = {}
        DatasetBestF2Train[dim][feature] = {}
        DatasetBestModelTrain[dim][feature] = {}
        Train_metrics[dim][feature] = {}
        

for file_name in files:
    field =  file_name[:-4].split("_")
    OutputPath = "./" + field[2] + "/" + field[3] + "/" + field[4]
    Path(OutputPath).mkdir(parents=True, exist_ok=True)

    Res = pd.read_csv(path +file_name, index_col=0).to_dict()
    ListOfConsideredMetrics = ["Recall", "Precision", "GMean", "F1Score", "F2Score", "F05Score", "LR+", "TPR", "FPR"]
    MetricsTraining = {}
    MetricsTest = {}
    for el in ListOfConsideredMetrics:
        MetricsTraining[el] = {}
        MetricsTest[el] = {}
    Test_metrics[int(field[2])][int(field[3])][int(field[4])] = {}
    Train_metrics[int(field[2])][int(field[3])][int(field[4])] = {}
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
        Test_metrics[int(field[2])][int(field[3])][int(field[4])][Thr] = M_Test[4]
        Train_metrics[int(field[2])][int(field[3])][int(field[4])][Thr] = M_Train[4]

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
  
    ROC[int(field[2])][int(field[3])][int(field[4])] = roc_score_training  
    PR[int(field[2])][int(field[3])][int(field[4])] = roc_pr_score_training       
    ROC_test[int(field[2])][int(field[3])][int(field[4])] = roc_score_test  
    PR_test[int(field[2])][int(field[3])][int(field[4])] = roc_pr_score_test
    #line = field[2] + r'&' + field[3] + r'&' + field[4] + r'&' +"{:.3f}".format(roc_score_training) + r'&' +"{:.3f}".format(roc_pr_score_training)  + r'&' +"{:.3f}".format(roc_score_test) + r'&' +"{:.3f}".format(roc_pr_score_test)+ r'\\' + "\n" + r' \hline' + " \n"
    #file_table.write(line)        
  
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
DataFramePandas.to_csv("NeuralNetwork.csv")



for dim in Dimensions:
    for feature in features:
        ROC_val = 0
        ROC_layer = 0
        PR_val = 0
        PR_layer = 0
        ROC_test_val = 0
        ROC_test_layer = 0
        PR_test_val = 0
        PR_test_layer = 0
        for key in ROC[dim][feature].keys():
            if ROC[dim][feature][key] > ROC_val or ROC_layer == 0:
                ROC_val = ROC[dim][feature][key]
                ROC_layer = key
            elif ROC[dim][feature][key] == ROC_val and ROC_layer != 0 and ROC_layer < key:
                ROC_val = ROC[dim][feature][key]
                ROC_layer = key
                
            if PR[dim][feature][key] > PR_val or PR_layer == 0:
                PR_val = PR[dim][feature][key]
                PR_layer = key
            elif PR[dim][feature][key] == PR_val and PR_layer != 0 and PR_layer < key:
                PR_val = PR[dim][feature][key]
                PR_layer = key
                
            if ROC_test[dim][feature][key] > ROC_test_val or ROC_test_layer == 0:
                ROC_test_val = ROC_test[dim][feature][key]
                ROC_test_layer = key
            elif ROC_test[dim][feature][key] == ROC_test_val and ROC_test_layer != 0 and ROC_test_layer < key:
                ROC_test_val = ROC_test[dim][feature][key]
                ROC_test_layer = key
                
            if PR_test[dim][feature][key] > PR_test_val or PR_test_layer == 0:
                PR_test_val = PR_test[dim][feature][key]
                PR_test_layer = key
            elif PR_test[dim][feature][key] == PR_test_val and PR_test_layer != 0 and PR_test_layer < key:
                PR_test_val = PR_test[dim][feature][key]
                PR_test_layer = key
                


        '''line = format(dim) + r'&' + format(feature) + r'&' + format(ROC_layer) + r'&' +"{:.3f}".format(ROC[dim][feature][ROC_layer]) + r'&' +"{:.3f}".format(PR[dim][feature][ROC_layer])  + r'&' +"{:.3f}".format(ROC_test[dim][feature][ROC_layer]) + r'&' +"{:.3f}".format(PR_test[dim][feature][ROC_layer])+ r'\\' + "\n" + r' \hline' + " \n"
        file_table.write(line)
        if ROC_layer != PR_layer:
            line = format(dim) + r'&' + format(feature) + r'&' + format(PR_layer) + r'&' +"{:.3f}".format(ROC[dim][feature][PR_layer]) + r'&' +"{:.3f}".format(PR[dim][feature][PR_layer])  + r'&' +"{:.3f}".format(ROC_test[dim][feature][PR_layer]) + r'&' +"{:.3f}".format(PR_test[dim][feature][PR_layer])+ r'\\' + "\n" + r' \hline' + " \n"
            file_table.write(line)
        if ROC_layer != ROC_test_layer and PR_layer != ROC_test_layer:
            line = format(dim) + r'&' + format(feature) + r'&' + format(ROC_test_layer) + r'&' +"{:.3f}".format(ROC[dim][feature][ROC_test_layer]) + r'&' +"{:.3f}".format(PR[dim][feature][ROC_test_layer])  + r'&' +"{:.3f}".format(ROC_test[dim][feature][ROC_test_layer]) + r'&' +"{:.3f}".format(PR_test[dim][feature][ROC_test_layer])+ r'\\' + "\n" + r' \hline' + " \n"
            file_table.write(line)
        if ROC_layer != PR_test_layer and PR_layer != PR_test_layer and ROC_test_layer != PR_test_layer:'''
        line = format(dim) + r'&' + format(feature) + r'&' + format(PR_test_layer) + r'&' +"{:.3f}".format(ROC[dim][feature][PR_test_layer]) + r'&' +"{:.3f}".format(PR[dim][feature][PR_test_layer])  + r'&' +"{:.3f}".format(ROC_test[dim][feature][PR_test_layer]) + r'&' +"{:.3f}".format(PR_test[dim][feature][PR_test_layer])+ r'\\' + "\n" + r' \hline' + " \n"
        file_table.write(line)
        BestTh = 0
        BestF2Score = 0       
        First = True     
        for thr in Test_metrics[dim][feature][PR_test_layer].keys():
            if First == True:
                BestTh = Thr
                BestF2Score = Test_metrics[dim][feature][PR_test_layer][thr]
                First = False                 
            elif Test_metrics[dim][feature][PR_test_layer][thr] > BestF2Score:
                BestTh = Thr
                BestF2Score = Test_metrics[dim][feature][PR_test_layer][thr]
        DatasetBestThreshold[dim][feature] = BestTh   
        DatasetBestF2[dim][feature] = BestF2Score
        DatasetBestModel[dim][feature] = PR_test_layer    

        BestThTrain = 0
        BestF2ScoreTrain = 0       
        First = True     
        for thr in Train_metrics[dim][feature][PR_layer].keys():
            if First == True:
                BestThTrain = Thr
                BestF2ScoreTrain = Train_metrics[dim][feature][PR_test_layer][thr]
                First = False                 
            elif Train_metrics[dim][feature][PR_layer][thr] > BestF2ScoreTrain:
                BestThTrain = Thr
                BestF2ScoreTrain = Train_metrics[dim][feature][PR_layer][thr]
        DatasetBestThresholdTrain[dim][feature] = BestThTrain   
        DatasetBestF2Train[dim][feature] = BestF2ScoreTrain
        DatasetBestModelTrain[dim][feature] = PR_layer        
            
DataFramePandas = pd.DataFrame.from_dict(DatasetBestThreshold, orient='index')
DataFramePandas.to_csv("BestThresholdNN.csv")
DataFramePandas.to_excel("BestThresholdNN.xlsx")
DataFramePandas = pd.DataFrame.from_dict(DatasetBestF2, orient='index')
DataFramePandas.to_csv("BestF2NN.csv")
DataFramePandas.to_excel("BestF2NN.xlsx")   
DataFramePandas = pd.DataFrame.from_dict(DatasetBestModel, orient='index')
DataFramePandas.to_csv("BestF2ModelNN.csv")
DataFramePandas.to_excel("BestF2ModelNN.xlsx")    


DataFramePandas = pd.DataFrame.from_dict(DatasetBestThresholdTrain, orient='index')
DataFramePandas.to_csv("BestThresholdNNTrain.csv")
DataFramePandas.to_excel("BestThresholdNNTrain.xlsx")
DataFramePandas = pd.DataFrame.from_dict(DatasetBestF2Train, orient='index')
DataFramePandas.to_csv("BestF2NNTrain.csv")
DataFramePandas.to_excel("BestF2NNTrain.xlsx")   
DataFramePandas = pd.DataFrame.from_dict(DatasetBestModelTrain, orient='index')
DataFramePandas.to_csv("BestF2ModelNNTrain.csv")
DataFramePandas.to_excel("BestF2ModelNNTrain.xlsx")         
               
file_table.write(r'\noalign{\hrule height 1.5pt}'+ "\n")
file_table.write(r'\end{tabular}' + "\n") 
file_table.write(r'\end{table}' + "\n")
file_table.close()
