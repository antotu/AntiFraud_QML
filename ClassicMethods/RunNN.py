import os
samples = [1000, 2000, 4000, 6000, 8000, 10000, 20000]
fH = [(2, [2, 3, 4, 5, 6]),(4, [2, 4, 6, 7, 8]), (6, [3, 5, 7, 8, 10]), (8, [3, 6, 8, 9, 11])] 
for s in samples:
    for f, H in fH:
        for h in H:
            os.system(f"python NeuralNetworks.py {s} {f} {h}")