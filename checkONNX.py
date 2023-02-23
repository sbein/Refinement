# import onnx
import onnxruntime
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile, MobileOptimizerType

import numpy as np


print(torch.__version__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trainingID = '20221127'

path_torch = '/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/regression_model_' + trainingID + '.pt'
path_onnx = '/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/regression_model_' + trainingID + '_opset11.onnx'

model_torch = torch.jit.load(path_torch)
model_torch.eval()
model_torch.to(device)

# onnx_model = onnx.load(path_onnx)
# onnx.checker.check_model(onnx_model)


x = torch.tensor([
    [331.162, 0.994019,  0.,  0.0215759,  0.945312, 0.380615,  0.141968],
    [17.7534, -0.269958,  0.,  0.00829315,  0.873047, 0.057373,  0.515137],
    [9.8506e+01, -7.7051e-01,  5.0000e+00,  9.9951e-01,  2.3007e-05, 6.5674e-01,  2.9907e-01]],
    requires_grad=True, device=device
)
torch_out = model_torch(x)


ort_session = onnxruntime.InferenceSession(path_onnx)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

print('input')
print(x)
print('torch out')
print(torch_out)
print('ort out')
print(ort_outs[0])

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


