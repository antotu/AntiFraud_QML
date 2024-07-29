import csv
import json
# read the csv file using csv reader
# discard the first line
# read the csv file using csv reader
Elements = []
with open('BestF2ModelNNTrain.csv', 'r') as file:
    csv_reader = csv.reader(file)
    
    # discard the first line
    next(csv_reader)
    
    # read each line of the file
    for line in csv_reader:
        for i in range(1, len(line)):
            Elements.append({"NumLayers": int(line[i]), "NumElements" : line[0], "NumFeatures" : i * 2})
            
# read the BestThesholdS.csv file
# discard the first line
with open("BestThresholdNNTrain.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    # read each line of the file
    for line in csv_reader:
        for i in range(1, len(line)):
            for el in Elements:
                if el["NumElements"] == line[0] and el["NumFeatures"] == i * 2:
                    el["Threshold"] = float(line[i])

# save Elements in a json file

with open("Elements_NNTrain.json", "w") as file:
    json.dump(Elements, file)