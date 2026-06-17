import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_torch_runtime():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return device
