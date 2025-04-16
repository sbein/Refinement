import torch
from torch.utils.mobile_optimizer import optimize_for_mobile, MobileOptimizerType

print(torch.__version__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trainingID = '20240717'
# trainingID = '20230112'

# path = '/afs/desy.de/user/w/wolfmor/cmssw/CMSSW_10_6_22/src/Refinement/training/'
# # model_path = path + 'dataloaderRegressionDummyONNX/weights/TrainedModel_PyTorch_Regression_Dummy_ONNX_20220509.pt'
# model_path = path + 'dataloaderRegression/weights/TrainedModel_PyTorch_Regression_20220513.pt'

model_path = '/data/dust/user/wolfmor/Refinement/TrainingOutput/model_refinement_regression_' + trainingID + '.pt'
out_path = model_path.replace('.pt', '_opset11.onnx')

model = torch.jit.load(model_path)

model.eval()


# model = model.cuda()

model.to(device)

model = optimize_for_mobile(model,
                            optimization_blocklist={MobileOptimizerType.INSERT_FOLD_PREPACK_OPS},
                            preserved_methods=['training'],
                            )

n_params = 3
n_vars = 3
batch_size = 1
x = torch.rand(batch_size, n_params+n_vars, requires_grad=True, device=device)
# x = torch.tensor([
#     [17.7534, -0.269958,  0.,  0.00829315,  0.873047, 0.057373,  0.515137],
#     [9.8506e+01, -7.7051e-01,  5.0000e+00,  9.9951e-01,  2.3007e-05, 6.5674e-01,  2.9907e-01]],
#     requires_grad=True, device=device
# )


out = model(x)


print(x)
print(out)


torch.onnx.export(model,                     # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    out_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    verbose=True,
                    training=torch.onnx.TrainingMode.EVAL,
                    input_names=['input'],     # the model's input names
                    output_names=['output'],   # the model's output names
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                    opset_version=11,  # 13,          # TODO: the ONNX version to export the model to, very sensitive... 9 is standard?
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    # example_outputs=out,  # only for v1.9 not for v1.11
                    # strip_doc_string=True,  # only for v1.9 not for v1.11
                    dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}},
                    keep_initializers_as_inputs=False,  # TODO: add this?
                    # enable_onnx_checker=True,  # only for v1.9 not for v1.11
                    )

print('just created ' + out_path)

