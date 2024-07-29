import pandas as pd
import math
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import matplotlib as mpl
from pathlib import Path
import numpy as np

rc('text', usetex=True)
plt.rc('text', usetex=True)

plt.rcParams.update({'font.size': 20})


def KLD(reference, device):
    D = 0
    for i in range(len(reference)):
        if device[i] != 0 and reference[i] != 0:
            D += reference[i] * math.log(
                reference[i] / device[i])
    return abs(D)


def HeligerFidelity(reference, device):
    temp = 0
    for i in range(len(reference)):
        temp += (math.sqrt(reference[i]) - math.sqrt(device[i])) ** 2
    H = (1 / math.sqrt(2)) * math.sqrt(temp)
    F = (1 - H ** 2) ** 2
    return F


Res = pd.read_csv("Result_comparison_Ideal_127.csv").to_dict()
Res2 = pd.read_csv("Result_comparison_27_127.csv").to_dict()

print(Res)
Samples = Res['Unnamed: 0']
p0_127 = Res['0_127']
p1_127 = Res['1_127']

p0_I = Res['0_Ideal']
p1_I = Res['1_Ideal']

Samples2 = Res2['Unnamed: 0']
p0_27 = Res2['0_127']
p1_27 = Res2['1_127']


Ideal = {}
D127 = {}
for i in range(len(Samples)):
    Ideal[Samples[i]] = [p0_I[i], p1_I[i]]
    D127[Samples[i]] = [p0_127[i], p1_127[i]]

D27 = {}
for i in range(len(Samples2)):
    D27[Samples[i]] = [p0_27[i], p1_27[i]]

FidelityI_127 = []
FidelityI_27 = []
KLDI_127 = []
KLDI_27 = []
for key in Ideal.keys():
    F = HeligerFidelity(Ideal[key], D127[key])
    K = KLD(Ideal[key], D127[key])
    FidelityI_127.append(F)
    KLDI_127.append(K)

for key in D27.keys():
    F = HeligerFidelity(Ideal[key], D27[key])
    K = KLD(Ideal[key], D27[key])
    FidelityI_27.append(F)
    KLDI_27.append(K)


print(len(FidelityI_27))
print(len(FidelityI_127))
sns.displot(FidelityI_127  )
#sns.displot(FidelityI_27)
plt.ylabel(r'\textbf{Counts}', fontsize=20)
plt.xlabel(r'\textbf{Fidelity}', fontsize=20)
plt.tight_layout()
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("FidelityI_127.eps", format='eps', bbox_inches='tight')
plt.savefig("FidelityI_127.png", format='png', bbox_inches='tight')
plt.savefig("FidelityI_127.pdf", format='pdf', bbox_inches='tight')
plt.close()

sns.displot(KLDI_127)
#sns.displot(FidelityI_27)
plt.ylabel(r'\textbf{Counts}', fontsize=20)
plt.xlabel(r'\textbf{KLD}', fontsize=20)
plt.tight_layout()
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("KLDI_127.eps", format='eps', bbox_inches='tight')
plt.savefig("KLDI_127.png", format='png', bbox_inches='tight')
plt.savefig("KLDI_127.pdf", format='pdf', bbox_inches='tight')
plt.close()




sns.displot(FidelityI_27 )
#sns.displot(FidelityI_27)
plt.ylabel(r'\textbf{Counts}', fontsize=20)
plt.xlabel(r'\textbf{Fidelity}', fontsize=20)
plt.tight_layout()
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("FidelityI_27.eps", format='eps', bbox_inches='tight')
plt.savefig("FidelityI_27.png", format='png', bbox_inches='tight')
plt.savefig("FidelityI_27.pdf", format='pdf', bbox_inches='tight')
plt.close()

sns.displot(KLDI_27 )
#sns.displot(FidelityI_27)
plt.ylabel(r'\textbf{Counts}', fontsize=20)
plt.xlabel(r'\textbf{KLD}', fontsize=20)
plt.tight_layout()
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("KLDI_27.eps", format='eps', bbox_inches='tight')
plt.savefig("KLDI_27.png", format='png', bbox_inches='tight')
plt.savefig("KLDI_27.pdf", format='pdf', bbox_inches='tight')
plt.close()

Fidelity = {}
Fidelity[r'\textit{ibm_brisbane}'] = FidelityI_127
Fidelity[r'\textit{ibm_algiers}'] = FidelityI_27

KLD = {}
KLD[r'\textit{ibm_brisbane}'] = KLDI_127
KLD[r'\textit{ibm_algiers}'] = KLDI_27



sns.displot(Fidelity, stat="probability", common_norm=False)
#sns.displot(FidelityI_27)
plt.ylabel(r'\textbf{Counts[\%]}', fontsize=20)
plt.xlabel(r'\textbf{Fidelity}', fontsize=20)
plt.tight_layout()
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("Fidelity.eps", format='eps', bbox_inches='tight')
plt.savefig("Fidelity.png", format='png', bbox_inches='tight')
plt.savefig("Fidelity.pdf", format='pdf', bbox_inches='tight')
plt.close()

sns.displot(KLD, stat="probability", common_norm=False)
#sns.displot(FidelityI_27)
plt.ylabel(r'\textbf{Counts[\%]}', fontsize=20)
plt.xlabel(r'\textbf{KLD}', fontsize=20)
plt.tight_layout()
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("KLD.eps", format='eps', bbox_inches='tight')
plt.savefig("KLD.png", format='png', bbox_inches='tight')
plt.savefig("KLD.pdf", format='pdf', bbox_inches='tight')
plt.close()