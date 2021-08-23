import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def forward(self, x):
        # first layer
        W1 = torch.Tensor([[1, 1], [1, 1]])
        b1 = torch.Tensor([0.0, 0.0])
        x = x @ W1 + b1
        x = F.relu(x)

        # second layer
        W2 = torch.Tensor([[1, 1], [1, 1]])
        b2 = torch.Tensor([0.0, 0.0])
        x = x @ W2 + b2
        x = F.relu(x)

        # third layer
        W3 = torch.Tensor([[3, -1], [4, 5]])
        b3 = torch.Tensor([0.0, 0.0])
        x = x @ W3 + b3
        x = F.relu(x)

        # output layer
        W4 = torch.Tensor([[-1, 1], [1, -1]])
        b4 = torch.Tensor([0.0, 0.0])
        x = x @ W4 + b4

        return x


# create instance of model
model = Model()
# export to onnx
torch.onnx.export(
    model,
    torch.randn((1, 2)),  # requires a dummy input value (can be anything)
    "sample_model_2.onnx",
    # next two lines aren't necessary for DNNV to work
    # but make first axis size dynamic (for input batches)
    # this is primarily useful if this was a model that you
    # were actually going to use
    input_names=["input"],
    dynamic_axes={"input": [0]},
)