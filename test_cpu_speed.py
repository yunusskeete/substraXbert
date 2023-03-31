import timeit
import torch
import random

x = torch.ones(5000, device="cpu")
print(timeit.timeit(lambda: x * random.randint(0,100), number=100000))