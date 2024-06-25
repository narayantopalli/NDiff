import cupy as cp
import ndiff.functions as F


class CompileFunc:
    def __init__(self, func, **kwargs):
        self.num_inputs, graph = func()
        if "show_graph" in kwargs:
            if kwargs["show_graph"]:
                func.show()
        self.graph = {}
        self.functions = F
        self.outputs = []
        self.xp = None
        for i in range(self.num_inputs):
            graph.append({'id': i, 'func': f"in_{i}", 'input_ids': [], 'constants': []})

        connected_nodes = set()
        for node in graph:
            for inp in node['input_ids']:
                if inp not in connected_nodes:
                    connected_nodes.add(inp)
            self.graph[node['id']] = {
                'func': node['func'],
                'input_ids': node['input_ids'],
                'constants': node['constants']
            }
        for i in range(len(graph)):
            if i not in connected_nodes:
                self.outputs.append(i)

        self.forward_pass = self._topological_sort()
        self.backward_pass = list(reversed(self.forward_pass))

    def _topological_sort(self):
        visited = set()
        topo_sorted = []

        def dfs(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            for inp_id in self.graph[node_id]['input_ids']:
                dfs(inp_id)
            topo_sorted.append(node_id)

        for id in self.graph:
            dfs(id)

        return topo_sorted

    def __call__(self, inputs, **kwargs):
        self.xp = cp.get_array_module(inputs)
        batch_size = inputs.shape[0]
        self.final_array = self.xp.empty((len(self.outputs), batch_size), dtype=self.xp.float32)
        self.jco = self.xp.empty((self.num_inputs, len(self.outputs), batch_size), dtype=self.xp.float32)
        return self.forward(inputs.T)

    def forward(self, inputs):
        evaluations = {}
        for id in self.forward_pass:
            if id < self.num_inputs:
                evaluations[id] = inputs[id]
            else:
                func = getattr(self.functions, self.graph[id]['func'])
                input_values = [evaluations[input_id] for input_id in self.graph[id]['input_ids']]
                for const in self.graph[id]['constants']:
                    input_values.insert(const[1], self.xp.array(const[0]))
                evaluations[id], self.graph[id]['grad'] = func(*input_values)

        for i, output in enumerate(self.outputs):
            self.final_array[i] = evaluations[output]
        return self.final_array.T

    def grad(self):
        gradients = {}

        for id in self.backward_pass:
            if id >= self.num_inputs:
                local_gradient = self.graph[id]['grad']

                if id in self.outputs:
                    outputs_arr = self.xp.zeros(len(self.outputs))
                    outputs_arr[self.outputs.index(id)] = 1.
                    new_grad = self.xp.tensordot(outputs_arr, local_gradient(1.), axes=0)
                else:
                    new_grad = local_gradient(1.)
                    new_grad = gradients[id][:, self.xp.newaxis, :] * new_grad[self.xp.newaxis, :, :]

                input_values = self.graph[id]['input_ids'][:]
                for const in self.graph[id]['constants']:
                    input_values.insert(const[1], -1)

                for i, input_id in enumerate(input_values):
                    if input_id == -1:
                        continue
                    elif input_id in gradients:
                        gradients[input_id] += new_grad.transpose(1, 0, 2)[i]
                    else:
                        gradients[input_id] = new_grad.transpose(1, 0, 2)[i]

        for i in range(self.num_inputs):
            self.jco[i] = gradients[i]

        return self.jco.transpose(2, 1, 0)
