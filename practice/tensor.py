import torch
import numpy as np

# Tensor creation 

# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From Numpy
np_array = np.array([[1, 2], [3, 4]])
x_np = torch.tensor(np_array)

# Same shape? -> torch.ones_like, torch.rand_like, torch.randn_like
x_ones = torch.ones_like(x_data)
"""
print(f"torch ones:\n")
print(x_ones)
print()
"""

x_rand = torch.rand_like(x_data, dtype = torch.float)
"""
print(f"torch random:\n")
print(x_rand)
print()
"""

# shape ? : a tuple that indicates a dimension of the tensor
shape = (2,3,)
rand_tensor = torch.rand(shape)
randn_tensor = torch.randn(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros((2,3,))

"""
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
"""

# Attribute of tensor
shape = (2,3)
tensor = torch.rand(shape)

"""
print(f"tensor shape: {tensor.shape}")
print(f"tensor dtype: {tensor.dtype}")
print(f"tensor device: {tensor.device}") # which device the tensor is stored on?
"""

# Tensor Operation

# Calculation of the tensor can be conducted faster when the tensor is on "GPU"
# If availabel move the tensor to GPU

"""
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"tensor is stored on {tensor.device}\n")
else:
    print(f"tensor is stored on {tensor.device}\n")
"""

# Slicing & Indexing
tensor = torch.ones((4,4))
# 2열만 0으로 바꾸기
tensor[:, 1] = 0.
"""
print(tensor)
"""

# Cat -> 갔다 붙인다.
cat = torch.cat([tensor, tensor, tensor], dim = 0) # (4,12)
"""
print(cat)
print(cat.shape)
"""

# element-wise multiply
"""
print(f"element wise mul: {tensor.mul(tensor)}\n")
print(f"element wise mul: {tensor * tensor}\n")
"""

# matrix multiply
"""
print(f"matmul : {tensor.matmul(tensor)}\n")
print(f"matmul : {tensor @ tensor}\n")
"""

# Convert tensor to numpy
shape = (5,)
t = torch.ones(shape)
print(f"t: {t}\n")
n = t.numpy()
print(f"n: {n}\n")

# Tensor의 변경 사항은 그대로 numpy에 반영됨.
t.add_(5) # "_"가 붙으면 in-place 연산임
print(f"t: {t}\n")
print(f"n: {n}\n")

# Numpy를 텐서로
num = np.ones(5)
ten = torch.from_numpy(num)

np.add(num, 1, out=num)
print(f"num: {num}\n")
print(f"ten: {ten}\n")

# Convert to python data from tensor
a = torch.ones((4,4))
sum = torch.sum(a)
sum_item = sum.item()

print(sum, sum_item)
print(f"sum_item dtype: {type(sum_item)}")