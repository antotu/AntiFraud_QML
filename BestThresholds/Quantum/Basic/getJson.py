import csv
import json
# read the csv file using csv reader
# discard the first line
# read the csv file using csv reader
Elements = []
with open('BestF2ModelBTrain.csv', 'r') as file:
    csv_reader = csv.reader(file)
    
    # discard the first line
    next(csv_reader)
    
    # read each line of the file
    for line in csv_reader:
        for i in range(1, len(line)):
            argL = line[i].split('_')
            reuploading = True if argL[-1] == "True" else False
            if len(argL) == 3:
                
                Elements.append({"Encoding": argL[1], "Ansatz": "Basic", "NumLayers": int(argL[0]), "Reuploading" : reuploading, "NumElements" : line[0], "NumFeatures" : i * 2})
            else:
                Elements.append({"Encoding": argL[1][:-1] + "_H", "Ansatz": "Basic", "NumLayers": int(argL[0]), "Reuploading" : reuploading, "NumElements" : line[0], "NumFeatures" : i * 2})

# read the BestThesholdS.csv file
# discard the first line
with open("BestThresholdBTrain.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    # read each line of the file
    for line in csv_reader:
        for i in range(1, len(line)):
            for el in Elements:
                if el["NumElements"] == line[0] and el["NumFeatures"] == i * 2:
                    el["Threshold"] = float(line[i])

# save Elements in a json file

with open("Elements_Train.json", "w") as file:
    json.dump(Elements, file)