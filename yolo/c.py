import torch
print(torch.version.cuda)       # Should print 11.8
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # Your GPU
