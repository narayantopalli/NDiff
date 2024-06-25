import numpy as np
import cupy as cp
import ndiff
import ndiff.functions as F
import timeit

@ndiff.Graph
def sigmoid(a):
    x = F.multiply(a, -1)
    x = F.exp(x)
    x = F.add(1, x)
    return F.divide(1, x)


compiled_func = ndiff.CompileFunc(sigmoid, show_graph=False)
inputs_gpu = cp.random.rand(1024, 1)
inputs_cpu = np.random.rand(1024, 1)

res = timeit.repeat(lambda: compiled_func(inputs_gpu), number=100)
print(f"GPU Best time: {min(res) * 1_000_000} microseconds")

res = timeit.repeat(lambda: compiled_func(inputs_cpu), number=100)
print(f"CPU Best time: {min(res) * 1_000_000} microseconds")


def grad_test(inputs):
    compiled_func(inputs)
    return compiled_func.grad()


res = timeit.repeat(lambda: grad_test(inputs_gpu), number=100)
print(f"GPU Grad Best time: {min(res) * 1_000_000} microseconds")

res = timeit.repeat(lambda: grad_test(inputs_cpu), number=100)
print(f"CPU Grad Best time: {min(res) * 1_000_000} microseconds")

res = compiled_func(cp.array([[0.75], [-1]]))
grad = compiled_func.grad()
print("forward:", res)
print("grad:", grad)
