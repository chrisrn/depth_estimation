import argparse
import os

import torch
import onnx
from src.UNet_plus_plus import UNet_plus_plus


def main(ckpt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = UNet_plus_plus().to(device)
    model.load_state_dict(ckpt["model_weights"])
    model.eval()

    example = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        scripted = torch.jit.trace(model, example)
    out_dir = "../triton_model_repo/depth_unet_ts/1"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    scripted.save(os.path.join(out_dir, "model.pt"))
    print("Saved TorchScript to model.pt")

    target_opset = min(19, onnx.defs.onnx_opset_version())
    torch.onnx.export(
        model, example, "model.onnx",
        input_names=["INPUT__0"], output_names=["OUTPUT__0"],
        opset_version=target_opset,
        dynamic_axes={"INPUT__0": {0: "B"}, "OUTPUT__0": {0: "B"}}
    )

    print("Saved onnx format")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, help='pytorch ckpt file')

    args = parser.parse_args()
    main(args.ckpt)