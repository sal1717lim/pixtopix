import torch
import utils
from resUnet import Generator
import config
model=Generator()
checkpoint = torch.load(r"C:\Users\SALIM\PycharmProjects\pixtopix\16gen.pth.tar", map_location=config.DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.pt')