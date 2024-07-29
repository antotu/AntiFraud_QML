import pennylane as qml
import torch
from torch import nn
from pennylane import numpy as np


class VQCStronglyEntanglingModel(nn.Module):
    """
    VQC 
    PARAMS:
    encoding: str, type of encoding chosen
    numLayerStrongly: int, number of layer for the strongly entangling layer ansatz
    numClass: int, to set the number of qubits to which the reading is applied
    numWires: int, set the number of wires of the quantum circuit
    gates: str, set the gates for the encoding
    reUploading: bool, to select if applying reuploading technique
    """

    def __init__(self, encoding: str,
                 numLayerStrongly: int,
                 numClasses: int,
                 numWires: int,
                 gates: str = "Y",
                 reUploading: bool =False):
        # initilize like the nn.Module class
        super().__init__()
        # set the parameters related to the Strongly Entangling layer part 
        shape = qml.StronglyEntanglingLayers.shape(n_layers=numLayerStrongly, n_wires=numWires)
        w1 = np.pi * np.random.random(shape, requires_grad=True)
        self.weightsStrongly = nn.Parameter(torch.from_numpy(w1), requires_grad=True)
        # set the encoding 
        self.encoding = encoding
        # set the gates applied
        self.gates = gates
        # set the number of layer for the strongly entangling circuit part
        self.numLayer = numLayerStrongly
        # set the reuploading 
        self.reUploading = reUploading
        # set the number of classes of the model
        self.numClasses = numClasses
        # set the number of wires
        self.num_wires = numWires
        # if the number of classes is larger than the number of wires
        # set the number of wires to the number of classes
        if numClasses > numWires:
            self.num_wires = self.numClasses

        self.bias = nn.Parameter(torch.from_numpy(np.zeros(1)), requires_grad=True)
    # superposition method
    # if called add for num_wires qubits the hadamard gates
    def superposition(self, num_wires):
        for i in range(num_wires):
            qml.Hadamard(wires=i)

    # set the encoding mechanism
    def set_encoding(self, features):
        # if the encoding used is the amplitude one
        # pad the vector with 0
        # and then apply the Mottonen State Preparation
        if self.encoding == "amplitude":
            features = features / np.linalg.norm(features)
            f = np.pad(features, (0, 2 ** self.num_wires - len(features)), "constant", constant_values=0)
            qml.MottonenStatePreparation(state_vector=f, wires=range(self.num_wires))
        # if angle encoding, apply the gates set by str
        elif self.encoding == "angle":
            possible_gate = ["X", "XY", "XYZ", "XZ", "XZY", "Y", "YX", "YXZ", "YZ", "YZX",
                             "Y_H", "YX_H", "YXZ_H", "YZ_H", "YZX_H", "Z_H", "ZX_H", "ZXY_H", "ZY_H", "ZYX_H"]
            gate_applied = self.gates
            if self.gates in possible_gate:
                # if _ is present, apply the hadamard gate
                if "_" in self.gates:
                    gate_applied = self.gates.split("_")[0]

                    self.superposition(num_wires=self.num_wires)
                # loop for each char of the string and apply the rotational gate 
                for rot in gate_applied:
                    cnt = 0
                    for f in features:
                        # if the char is X, apply a RX gate, if Y RY else if Z RZ
                        if rot == "X":
                            qml.RX(f * np.pi / 2, wires=cnt)
                        elif rot == "Y":
                            qml.RY(f * np.pi / 2, wires=cnt)
                        else:
                            qml.RZ(f * np.pi / 2, wires=cnt)
                        cnt += 1
            else:
                print("Error in the encoding gates")
                exit(1)
        else:
            print(self.gates)
            print("Error in the definition of the encoding")
            exit(-1)

    # set the number of measured qubits
    def set_num_meas_wire(self):
        # if binary classification, read just the first qubit
        if self.numClasses == 2:
            return 1
        else:
            # else read the same number of qubits of the
            # number of classes of the dataset 
            return self.numClasses

  

    # describe the quantum circuit
    def circuit(self, features):
        # check if reuploading variable is true
        if self.reUploading:
            # in this case apply reuploading for each ansatz layer
            for i in self.weightsStrongly:
                self.set_encoding(features=features)
                qml.StronglyEntanglingLayers(i.unsqueeze(0), wires=range(self.num_wires))
        else:
            # otherwise apply just encoding and ansatz
            self.set_encoding(features=features)
            qml.StronglyEntanglingLayers(self.weightsStrongly, wires=range(self.num_wires))

        return [qml.expval(qml.PauliZ(wires=[self.num_wires - j - 1]))
                for j in range(self.set_num_meas_wire())]

    # the forward method consist of saving the output of the circuit
    def forward(self, X):
        # define the characteristics of the device
        # the default number of shots is equal to 1000 for each circuit
        dev = qml.device("default.qubit", wires=self.num_wires, shots=10000)
        vqc = qml.QNode(self.circuit, dev, interface="torch")
        res = []
        for x in X:
            res.append(vqc(x) + self.bias)
        res = torch.stack(res)
        return res
