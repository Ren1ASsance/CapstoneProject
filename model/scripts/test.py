import torch
print(torch.cuda.is_available())  # 如果返回 True，说明可以使用 GPU
print(torch.cuda.current_device())  # 显示当前使用的 GPU 设备
print(torch.cuda.get_device_name(0))  # 显示 GPU 名称
