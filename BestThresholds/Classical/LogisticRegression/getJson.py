import csv
import json
# read the csv file using csv reader
# discard the first line
# read the csv file using csv reader
Elements = []
with open("BestThresholdLogisticRegressionTrain.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    # read each line of the file
    for line in csv_reader:
        for i in range(1, len(line)):
            el = {"NumElements": line[0], "NumFeatures" : i * 2, "Threshold" : line[i]}
            print(el)
            Elements.append(el)
    

# save Elements in a json file

with open("Elements_LogisticRegressionTrain.json", "w") as file:
    json.dump(Elements, file)