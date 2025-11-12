import numpy as np, cv2
from PIL import Image
import tritonclient.http as http

MODEL_NAME = "depth_unet_ts"
H, W = 224, 224

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((W, H))
    x = np.array(img).astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))
    return x[None, ...]

triton = http.InferenceServerClient(url="localhost:8000")
x = preprocess("00034_colors.png")

inputs = [http.InferInput("INPUT__0", x.shape, "FP32")]
inputs[0].set_data_from_numpy(x, binary_data=False)

outputs = [http.InferRequestedOutput("OUTPUT__0", binary_data=False)]
resp = triton.infer(MODEL_NAME, inputs=inputs, outputs=outputs)
y = resp.as_numpy("OUTPUT__0")
y = y.squeeze()
y_norm = ((y - y.min()) / (y.ptp() + 1e-8) * 255).astype(np.uint8)
cv2.imwrite("depth_pred.png", y_norm)
print("Saved depth_pred.png")
