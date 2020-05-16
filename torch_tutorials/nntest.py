import torch

grads = {}
grads_list = [0]

def save_grad(grad_name):
    def hook(grad):
        grads[grad_name] = grad
    return hook

x1 = torch.ones(2, 2, requires_grad=False)
x2 = torch.randn(2, 2, requires_grad=True)
# 当有一个变量requires_grad=True, 就可以构成计算图，反向计算
y = x1 + x2 +2
z = y * y * 3
out = z.mean()
y.register_hook(save_grad("y"))
z.register_hook(save_grad("z"))
out.register_hook(save_grad("out"))

print("z = \n", z)
print("out = \n", out)

out.backward()


print("x1.grad = \n", x1.grad)
print("x2.grad = \n", x2.grad)
print("y.grad = \n", y.grad)
print("grads[z] = \n",  grads["z"])
print("z.grad = \n", z.grad)
print("grads[z] = \n", grads["z"])
print("grads[out] = \n", grads["out"])
print("out.grad = \n", out.grad)
