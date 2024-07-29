import pandas as pd
import math
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
import numpy as np
from sklearn.metrics import auc



rc('text', usetex=True)
plt.rc('text', usetex=True)

plt.rcParams.update({'font.size': 20})

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

AreaBelowCurves = {}

   
#Dimensions = [1000, 2000, 4000, 6000, 8000, 10000, 20000]
#features = [2, 4, 6, 8]

ROC = 0
PR = 0
ROC_test = 0
PR_test = 0

threas = [i/1000 for i in range(1, 100)] + [i/10 for i in range(1, 6)]

Metrics = {}

Res = pd.read_csv("Result_comparison_Ideal_127.csv").to_dict()

PredictionIdeal = list(Res["prediction_Ideal"].values())

Prediction127 = list(Res["prediction_127"].values())

Labels = list(Res["label"].values())

TPI = {}
TNI = {}
FPI = {}
FNI = {}

TP127 = {} 
TN127 = {}
FP127 = {}
FN127 = {}

for th in threas:
    TPI[th] = 0
    TNI[th] = 0
    FPI[th] = 0
    FNI[th] = 0

    TP127[th] = 0
    TN127[th] = 0
    FP127[th] = 0
    FN127[th] = 0
    for i in range(len(Labels)): 
        if Labels[i] == 1:
            if PredictionIdeal[i] >= th:
                TPI[th] += 1
            else: 
                FNI[th] += 1
                
            if Prediction127[i] >= th:
                TP127[th] += 1
            else: 
                FN127[th] += 1
        else:
            if PredictionIdeal[i] >= th:
                FPI[th] += 1
            else: 
                TNI[th] += 1
                
            if Prediction127[i] >= th:
                FP127[th] += 1
            else: 
                TN127[th] += 1
            
DicRes = {}
DicRes["TPI"] = TPI
DicRes["TNI"] = TNI
DicRes["FPI"] = FPI
DicRes["FNI"] = FNI 

DicRes["TP127"] = TP127
DicRes["TN127"] = TN127
DicRes["FP127"] = FP127
DicRes["FN127"] = FN127

print(DicRes)

DataFramePandas = pd.DataFrame.from_dict(DicRes, orient='index')
DataFramePandas.to_csv("ValuesTh.csv")

ListOfConsideredMetrics = ["Recall", "Precision", "GMean", "F1Score", "F2Score", "F05Score", "LR+", "TPR", "FPR"]
MetricsI = {}
Metrics127 = {}
for el in ListOfConsideredMetrics:
    MetricsI[el] = {}
    Metrics127[el] = {}

Metrics = {}

for th in threas: 

    TPI = DicRes["TPI"][th]
    TNI = DicRes["TNI"][th]
    FPI = DicRes["FPI"][th]
    FNI = DicRes["FNI"][th]

    TP127 = DicRes["TP127"][th]
    TN127 = DicRes["TN127"][th]
    FP127 = DicRes["FP127"][th]
    FN127 = DicRes["FN127"][th]
    
    MI = compute_all(TPI, TNI, FPI, FNI)
    M127 = compute_all(TP127, TN127, FP127, FN127)
    
    for j in range(len(ListOfConsideredMetrics)):
        MetricsI[ListOfConsideredMetrics[j]][th] = MI[j]
        Metrics127[ListOfConsideredMetrics[j]][th] = M127[j]
        
DataFramePandas = pd.DataFrame.from_dict(MetricsI, orient='index')
DataFramePandas.to_csv("metricsI.csv")
DataFramePandas = pd.DataFrame.from_dict(Metrics127, orient='index')
DataFramePandas.to_csv("metrics127.csv")

AreaBelowCurves = {}
roc_scoreI= auc(list(MetricsI["FPR"].values()), list(MetricsI["TPR"].values()))
AreaBelowCurves["roc_scoreI"] = roc_scoreI

roc_score127 = auc(list(Metrics127["FPR"].values()), list(Metrics127["TPR"].values()))
AreaBelowCurves["roc_score127"] = roc_score127

roc_pr_scoreI = auc(list(MetricsI["Recall"].values()), list(MetricsI["Precision"].values()))
AreaBelowCurves["roc_pr_scoreI"] = roc_pr_scoreI

roc_pr_score127 = auc(list(Metrics127["Recall"].values()), list(Metrics127["Precision"].values()))
AreaBelowCurves["roc_pr__score127"] = roc_pr_score127

for k in range(len(ListOfConsideredMetrics)):
    plt.plot(list(MetricsI[ListOfConsideredMetrics[k]].keys()),list(MetricsI[ListOfConsideredMetrics[k]].values()), linewidth=2, label=r'\textit{Ideal Simulation}')
    plt.plot(list(Metrics127[ListOfConsideredMetrics[k]].keys()),list(Metrics127[ListOfConsideredMetrics[k]].values()), linewidth=2, label=r'\textit{Real Device}')
    plt.xlabel(r'\textbf{Thresholds}', fontsize=20)
    plt.ylabel(r'\textbf{'+ ListOfConsideredMetrics[k] +'}', fontsize=20)
    plt.legend(loc='center right')
    file_name = str(ListOfConsideredMetrics[k]) + ".pdf"
    print(file_name)
    plt.savefig(file_name, format='pdf', bbox_inches='tight')
    plt.close()
    
    
plt.plot(list(MetricsI["Recall"].values()), list(MetricsI["Precision"].values()),  label=r'\textit{Ideal Simulation}', linewidth=2)
plt.plot(list(Metrics127["Recall"].values()), list(Metrics127["Precision"].values()),  label=r'\textit{Real Device}', linewidth=2)
plt.xlabel(r'\textbf{Recall}', fontsize=20)
plt.ylabel(r'\textbf{Precision}', fontsize=20)
plt.legend(loc='center right')
#plt.savefig(OutputPath + "/" + file_name[:-4] + "_RecallPrecisionTraining" + ".eps", format='eps',bbox_inches='tight')
plt.savefig("RecallPrecision.png", format='png',bbox_inches='tight')
plt.savefig("RecallPrecision.pdf",format='pdf',bbox_inches='tight')
plt.close()

plt.plot(list(MetricsI["FPR"].values()), list(MetricsI["TPR"].values()), label=r'\textit{Ideal Simulation}', linewidth=2)
plt.plot(list(Metrics127["FPR"].values()), list(Metrics127["TPR"].values()), label=r'\textit{Real Device}', linewidth=2)
plt.ylabel(r'\textbf{True Positive Ratio}', fontsize=20)
plt.xlabel(r'\textbf{False Positive Ratio}', fontsize=20)
plt.legend(loc='lower right')
#plt.savefig(OutputPath + "/" + file_name[:-4] + "_TrueFalsePositiveRatioTraining" + ".eps", format='eps',bbox_inches='tight')
plt.savefig("TrueFalsePositiveRatio.png", format='png',bbox_inches='tight')
plt.savefig("TrueFalsePositiveRatio.pdf", format='pdf',bbox_inches='tight')
plt.close()


DataFramePandas = pd.DataFrame.from_dict(AreaBelowCurves, orient='index')
DataFramePandas.to_csv("AreaBelowCurves.csv")
