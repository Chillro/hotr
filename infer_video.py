# infer_video.py
# HOTR inference on a single mp4 (no dataset eval). Triplets are generated via the same postprocessor as main.

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from PIL import Image

# --- make sure repo root is on sys.path so `import hotr` works regardless of cwd ---
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hotr.engine.arg_parser import get_args_parser
from hotr.models import build_model
from hotr.util.misc import nested_tensor_from_tensor_list
from hotr.data.datasets.builtin_meta import _get_coco_instances_meta


def parse_cli():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--video", required=True)
    p.add_argument("--resume", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out_dir", default=r"C:\Users\takeu\pydata\hotr\hotr\outputs")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--score_thresh", type=float, default=0.0)
    p.add_argument("--log_interval", type=int, default=10,
                   help="print progress every N processed frames")
    return p.parse_args()


def build_hotr_args_for_hico(device: str):
    args = get_args_parser().parse_args([])

    args.HOIDet = True
    args.share_enc = True
    args.pretrained_dec = True
    args.num_hoi_queries = 16
    args.temperature = 0.2
    args.no_aux_loss = True
    args.eval = True

    args.num_classes = 91
    args.num_actions = 117
    args.action_names = None
    args.valid_obj_ids = list(range(91))
    args.object_threshold = 0.0

    args.device = device
    args.dataset_file = "hico-det"

    return args


def default_transform():
    return T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def try_extract_triplets(postprocessed, score_thresh: float):
    triplets = []

    # postprocessed is usually a list (batch size = 1)
    if isinstance(postprocessed, list) and len(postprocessed) > 0:
        postprocessed = postprocessed[0]

    # must be dict
    if not isinstance(postprocessed, dict):
        return triplets

    # HICO-DET / HOTR output key
    if "verb_scores" not in postprocessed:
        return triplets

    vs = postprocessed["verb_scores"]    # shape: [Q, V]
    sub_ids = postprocessed["sub_ids"]   # shape: [Q]
    obj_ids = postprocessed["obj_ids"]   # shape: [Q]

    Q, V = vs.shape

    for i in range(Q):
        sub_id = int(sub_ids[i])
        obj_id = int(obj_ids[i])

        for verb_id in range(V):
            score = float(vs[i, verb_id])
            if score >= score_thresh:
                triplets.append({
                    "sub_id": sub_id,
                    "obj_id": obj_id,
                    "verb_id": verb_id,
                    "score": score,
                })

    return triplets

def decode_triplet(triplet, coco_classes, verb_names, obj_labels):
    subject = "human"

    obj_box_id = triplet["obj_id"]   # box index
    verb_id    = triplet["verb_id"]

    # box index → class id
    if obj_box_id < len(obj_labels):
        class_id = int(obj_labels[obj_box_id])
        obj_name = coco_classes[class_id]
    else:
        obj_name = f"obj_{obj_box_id}"

    if verb_id < len(verb_names):
        verb_name = verb_names[verb_id]
    else:
        verb_name = f"verb_{verb_id}"
    if verb_name == "no_interaction":
        return None

    return {
        "subject": subject,
        "verb":    verb_name,
        "object":  obj_name,
        "score":   triplet["score"],
    }

def draw_boxes(frame, boxes, labels=None, color=(0, 255, 0)):
    """
    frame  : np.ndarray (BGR)
    boxes  : Tensor[N, 4] in xyxy
    labels : list[str] or None
    """
    h, w = frame.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int().tolist()

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        if labels is not None:
            cv2.putText(
                frame,
                labels[i],
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

def main():
    # === label tables ===
    coco_meta = _get_coco_instances_meta()
    COCO_CLASSES = coco_meta["coco_classes"]

    # load verb names from list_action.txt
    action_list = Path(r"C:\Users\takeu\pydata\hotr\hico_20160224_det\list_action.txt")
    verb_names = []
    with open(action_list, "r") as f:
        for line in f.readlines()[2:]:
            _, name = line.split()
            verb_names.append(name)

    cli = parse_cli()

    video_path = Path(cli.video)
    ckpt_path = Path(cli.resume)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = torch.device(
        cli.device if (cli.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    args = build_hotr_args_for_hico(str(device))
    model, _, postprocessors = build_model(args)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    hoi_pp = None
    if isinstance(postprocessors, dict):
        for v in postprocessors.values():
            hoi_pp = v
            break

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    jsonl_path = out_dir / "triplets.jsonl"

    processed = 0
    frame_idx = -1
    start_time = time.time()  

    transform = default_transform()

    with open(jsonl_path, "w", encoding="utf-8") as w:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % cli.stride != 0:
                continue

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = transform(img)
            nested = nested_tensor_from_tensor_list([tensor]).to(device)

            h, w_ = frame.shape[:2]
            target_sizes = torch.tensor([[h, w_]], device=device)

            with torch.no_grad():
                outputs = model(nested)

            pp = hoi_pp(outputs, target_sizes, args.object_threshold, args.dataset_file)

            pp0 = pp[0]   # batch size = 1
            boxes = pp0["boxes"]          # Tensor [N,4]
            sub_ids = pp0["sub_ids"]      # Tensor [Q]
            obj_ids = pp0["obj_ids"]      # Tensor [Q]
            obj_labels = pp0["labels"]   # Tensor [N] (COCO class id, contiguous)

            # object class names
            obj_names = [
                COCO_CLASSES[int(cid)]
                if int(cid) < len(COCO_CLASSES) else f"obj_{int(cid)}"
                for cid in obj_labels
            ]
    
            frame_vis = frame.copy()

            # subject (human) boxes : green
            draw_boxes(
                frame_vis,
                boxes[sub_ids],
                labels = ["human"] * len(sub_ids),
                color  = (0, 255, 0)
            )

            # object boxes : red
            draw_boxes(
                frame_vis,
                boxes[obj_ids],
                labels = [obj_names[i] for i in obj_ids],
                color  = (0, 0, 255)
            )

            frame_out_dir = out_dir / "frames"
            frame_out_dir.mkdir(exist_ok=True)

            cv2.imwrite(
                str(frame_out_dir / f"frame_{frame_idx:06d}.jpg"),
                frame_vis
            )

            raw_triplets = try_extract_triplets(pp, cli.score_thresh)

            decoded_triplets = [
                d for d in (
                    decode_triplet(t, COCO_CLASSES, verb_names, obj_labels)
                    for t in raw_triplets
                )
                if d is not None
            ]
            
            if fps and fps > 0:
                print("=" * 60)
                print(f"[DEBUG] frame {frame_idx} ({frame_idx / fps:.2f} sec)")

                if "pred_actions" in outputs:
                    print(
                        "pred_actions stats:",
                        outputs["pred_actions"].min().item(),
                        outputs["pred_actions"].max().item()
                    )
                else:
                    print("pred_actions: NOT FOUND")

                print(f"triplets count: {len(decoded_triplets)}")
            if len(decoded_triplets) > 0:
                topk = sorted(
                    decoded_triplets,
                    key=lambda x: x["score"],
                    reverse=True
                )[:5]

                for rank, t in enumerate(topk, 1):
                    print(
                        f"[TOP-{rank}] "
                        f"{t['subject']} {t['verb']} {t['object']} "
                        f"(score={t['score']:.4e})"
                    )
            else:
                print("no triplets detected")

            print("=" * 60)

            w.write(json.dumps({
                "frame_idx": frame_idx,
                "time_sec": frame_idx / fps if fps > 0 else None,
                "triplets": decoded_triplets,
            }) + "\n")

            processed += 1

            # === progress log ===
            if processed % cli.log_interval == 0:
                elapsed = time.time() - start_time
                avg_fps = processed / elapsed if elapsed > 0 else 0.0
                percent = (frame_idx + 1) / total_frames * 100
                eta = (total_frames - frame_idx - 1) / avg_fps if avg_fps > 0 else 0.0

                print(
                    f"[Progress] "
                    f"{frame_idx+1}/{total_frames} "
                    f"({percent:5.1f}%) | "
                    f"processed={processed} | "
                    f"avg_fps={avg_fps:5.2f} | "
                    f"ETA={eta:6.1f}s"
                )

            if cli.max_frames > 0 and processed >= cli.max_frames:
                break

    cap.release()

    print(f"[DONE] {processed} frames processed")
    print(f"[DONE] output: {jsonl_path}")


if __name__ == "__main__":
    main()
