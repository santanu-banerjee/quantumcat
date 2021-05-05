# (C) Copyright Artificial Brain 2021.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from quantumcat.circuit import QCircuit
import numpy as np
from quantumcat.utils import helper
from quantumcat.utils import providers


class ClassifierCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits):
        super(ClassifierCircuit, self).__init__()
        self.circuit = QCircuit(n_qubits, n_qubits)
        self.theta = ''
        self.all_qubits = [i for i in range(n_qubits)]

    def N_qubit_expectation_Z(self, counts, shots, nr_qubits):
        expects = np.zeros(nr_qubits)
        for key in counts.keys():
            perc = counts[key]/shots
            check = np.array([(float(key[i])-1/2)*2*perc for i in range(nr_qubits)])
            expects += check
        return expects

    def bind(self, parameters):
        [self.theta] = helper.to_numbers(parameters)

        for qubit in self.all_qubits:
            self.circuit.h_gate(qubit)
            self.circuit.ry_gate(self.theta, qubit)
            self.circuit.measure(qubit, qubit)

    def run(self, i):
        self.bind(i)
        rep = 1000
        counts = self.circuit.execute(provider = providers.IBM_PROVIDER, repetitions=rep)
        return self.N_qubit_expectation_Z(counts, rep, 1)