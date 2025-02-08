import torch
from torch import nn

torch.manual_seed(1)

x = torch.ones(5)
y = torch.zeros(3)
W = torch.randn((5,3), requires_grad = True)
b = torch.randn(3, requires_grad = True)
z = x @ W + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# Shows how each tensor has been made by
print(f"loss : {loss.grad_fn}\n")
print(f"z : {z.grad_fn}\n")

# Backward for grad calculation for W, b of loss
loss.backward()

# Grad values
print(W.grad)
print()
print(b.grad)

# 연산 기록 추적 중지 및 변화도 계산 지원 x
z = x @ W + b
print(z.requires_grad)
print()

with torch.no_grad(): # 이 구문 안에서 일어나는 연산은 기울기 추적이 일어나지 않음.
    z = x @ W + b

print(z.requires_grad)
print()

z = x @ W + b
z_det = z.detach() # detach는 기울기 추적을 차단한 복사본 텐서를 생성
print(z_det.requires_grad)
print(z.requires_grad)