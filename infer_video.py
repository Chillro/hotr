import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from hotr.models import build_model
from hotr.util.misc import nested_tensor_from_tensor_list

# ========= 設定 =========
VIDEO_PATH = "inputs/my_video.mp4"
CKPT_PATH  = "checkpoints/hico_det/hico_16.pth"
DEVICE = "cpu"   # cuda 可

# ========= transform =========
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ========= 最小 args =========
class Args:
    backbone="resnet50"
    dilation=False
    position_embedding="sine"
    enc_layers=6
    dec_layers=6
    dim_feedforward=2048
    hidden_dim=256
    dropout=0.1
    nheads=8
    num_queries=100
    masks=False
    aux_loss=False

    HOIDet=True
    share_enc=True
    pretrained_dec=True
    hoi_enc_layers=6
    hoi_dec_layers=6
    hoi_nheads=8
    hoi_dim_feedforward=2048
    num_hoi_queries=16
    hoi_aux_loss=False
    temperature=0.2

args = Args()

# ========= model =========
model, _, _ = build_model(args)
ckpt = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)
model.to(DEVICE).eval()

# ========= video =========
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img)
    nested = nested_tensor_from_tensor_list([tensor]).to(DEVICE)

    with torch.no_grad():
        outputs = model(nested)

    # ここで outputs から HOI triplet を取り出す
    print(outputs.keys())

cap.release()
