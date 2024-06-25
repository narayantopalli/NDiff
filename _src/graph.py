import numpy as np
import ndiff.functions
import cupy as cp
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, func):
        self.func = func
        self.functions = ndiff.functions
        self.primitives = ['sin', 'cos', 'exp', 'add', 'subtract',
                           'multiply', 'divide', 'power', 'maximum', 'minimum']
        self.original_functions = {}
        self.graph = []
        self.counter = 0
        self._inputs = []
        self.xp = None

        # Original functions
        for primitive in self.primitives:
            self.original_functions[primitive] = getattr(self.functions, primitive)
            setattr(self.functions, primitive, self._create_traced_function(primitive))

    def __call__(self, device='cpu', **kwargs):
        input_count = self.func.__code__.co_argcount
        if device == 'cuda':
            inputs = cp.random.rand(input_count)
        else:
            inputs = np.random.rand(input_count)

        self.xp = cp.get_array_module(inputs)
        for i in range(len(inputs)):
            self._inputs.append([inputs[i].tolist(), i])
            self.counter += 1

        res = self.func(*self._inputs, **kwargs)
        for primitive in self.primitives:
            setattr(self.functions, primitive, self.original_functions[primitive])
        return len(self._inputs), self.graph

    def _create_traced_function(self, func_name):
        def traced_function(*args):
            return self._trace_function(func_name, [*args])

        return traced_function

    def _trace_function(self, func_name, x):
        constants = []
        new_x = []
        for i in range(len(x)):
            if type(x[i]) is not list:
                constants.append((x[i], i))
            else:
                new_x.append(x[i])
        inputs, indexes = self.xp.array(new_x).T
        inputs = list(inputs)
        for const in constants:
            inputs.insert(const[1], self.xp.array(const[0]))
        res, _ = self.original_functions[func_name](*inputs)
        id = self.counter
        trace = {
            'id': id,
            'func': func_name,
            'input_ids': indexes.astype(self.xp.int32).tolist(),
            'constants': constants
        }
        self.counter += 1
        self.graph.append(trace)
        return [res.tolist(), id]

    def __str__(self):
        G = nx.DiGraph()
        for node in self.graph:
            G.add_node(node['id'], label=f"{node['func']} {', '.join([str(const[0]) for const in node['constants']])}")
            for input_id in node['input_ids']:
                G.add_edge(input_id, node['id'])
        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color='skyblue')
        plt.show()
        return "Printed Graph"
