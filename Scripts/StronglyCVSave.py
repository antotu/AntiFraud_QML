import os
import sys

encAngle = ["X", "XY", "XYZ", "XZ", "XZY", "Y", "YX", "YXZ", "YZ", "YZX", "Y_H", "YX_H", "YXZ_H", "YZ_H", "YZX_H", "Z_H", "ZX_H", "ZXY_H", "ZY_H", "ZYX_H"]
"""if en == "X":
    encAngle = ["X", "XY", "XYZ", "XZ", "XZY"]
elif en == "Y":
    encAngle = ["Y", "YX", "YXZ", "YZ", "YZX"]
elif en == "Y_H":
    encAngle = ["Y_H", "YX_H", "YXZ_H", "YZ_H", "YZX_H"]
elif en == "Z_H":
    encAngle = ["Z_H", "ZX_H", "ZXY_H", "ZY_H", "ZYX_H"]
else:
    print("Error")
    exit(1)
"""
numFeatures = [6, 8]
numElements = [1000, 2000, 4000, 6000, 8000, 10000, 20000]
numLayers = [2, 4, 6, 8, 10]
reuploading = [0, 1]

for f in numFeatures:
    for e in numElements:
        for l in numLayers:
            for a in encAngle:
                for r in reuploading:
                    print(f"python FraudDetection_ClassificationStrongly.py {e} {f} {a} {l} {r}")
                    os.system(f"python FraudDetection_ClassificationStrongly.py {e} {f} {a} {l} {r}")