import os
samples = [1000, 2000, 4000, 6000, 8000, 10000, 20000]
features = [2, 4, 6, 8]
kernel = ["rbf", "linear"]
for k in kernel:
    for s in samples:
        for f in features:
            os.system(f"python SVMModel.py {s} {f} {k}")