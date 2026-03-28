"""
Child Safety Pipeline - Detect when children stray from their group.

Usage:
  python child_safety_demo.py --source video.mp4 --output result.mp4
  python child_safety_demo.py --source 0  # webcam
"""

import argparse
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

sys.path.insert(0, '/tmp/MiVOLO')

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster, linkage
from transformers import AutoModelForImageClassification, AutoConfig
from ultralytics import YOLO
from torchvision import transforms
import timm

CHILD_AGE_THRESHOLD = 16
CHILD_HEIGHT_RATIO = 0.80   # if person bbox height < 80% of tallest nearby person, likely a child
GROUP_DISTANCE_THRESHOLD = 200
STRAY_DISTANCE_THRESHOLD = 150
AGE_INTERVAL = 5
AGE_HISTORY_LEN = 10
HISTORY_LEN = 30


def load_mivolo(device='cuda'):
    model = AutoModelForImageClassification.from_pretrained(
        '/home/cahlen/dev/gguf-workbench/mivolo-v2-finetuned', trust_remote_code=True,
    ).float().half().to(device)
    model.requires_grad_(False)
    return model


def load_child_classifier(device='cuda'):
    ckpt = torch.load('/home/cahlen/dev/gguf-workbench/child-classifier/child_classifier.pt', map_location='cpu')
    model_name = ckpt.get('model_name', 'mobilenetv3_large_100')
    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).half()
    model.requires_grad_(False)
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf


def find_face_for_person(person_bbox, face_boxes):
    """Find the face box that's inside a person box."""
    px1, py1, px2, py2 = person_bbox
    best_face = None
    best_overlap = 0
    for fb in face_boxes:
        fx1, fy1, fx2, fy2 = fb
        # Face center should be inside person box
        fcx, fcy = (fx1+fx2)/2, (fy1+fy2)/2
        if px1 <= fcx <= px2 and py1 <= fcy <= py2:
            area = (fx2-fx1) * (fy2-fy1)
            if area > best_overlap:
                best_overlap = area
                best_face = fb
    return best_face


def crop_and_resize(frame, bbox, size=384):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), min(w, int(bbox[2])), min(h, int(bbox[3]))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


def get_age(model, frame, person_bbox, face_bbox=None, device='cuda'):
    body_crop = crop_and_resize(frame, person_bbox)
    if body_crop is None:
        return None
    body = torch.from_numpy(body_crop).permute(2, 0, 1).float() / 255.0

    if face_bbox is not None:
        face_crop = crop_and_resize(frame, face_bbox)
        if face_crop is not None:
            face = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
        else:
            face = torch.zeros(3, 384, 384)
    else:
        face = torch.zeros(3, 384, 384)

    inp = torch.cat([face, body], dim=0).unsqueeze(0).half().to(device)
    with torch.no_grad():
        out = model(concat_input=inp)
    return out.age_output.item()


def cluster_groups(positions_dict, threshold=GROUP_DISTANCE_THRESHOLD):
    if len(positions_dict) < 2:
        return {tid: 0 for tid in positions_dict}
    tids = list(positions_dict.keys())
    pts = np.array([positions_dict[t] for t in tids])
    Z = linkage(pdist(pts), method='average')
    labels = fcluster(Z, t=threshold, criterion='distance')
    return dict(zip(tids, [int(l) for l in labels]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='0')
    parser.add_argument('--output', default='/home/cahlen/dev/gguf-workbench/pipeline_output.mp4')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    print("Loading YOLO + MiVOLO + Child Classifier...")
    yolo = YOLO('/home/cahlen/dev/gguf-workbench/yolo-face-person/yolov8x_person_face.pt')
    mivolo = load_mivolo()
    child_clf, child_tf = load_child_classifier()
    print("Ready.")

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open {args.source}")
        return

    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(5) or 30
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) if args.output else None

    ages = {}
    age_readings = defaultdict(list)
    clf_history = defaultdict(list)   # track_id -> list of recent child probs
    history = defaultdict(list)
    groups = {}
    fnum = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fnum += 1

        # Detect persons (class 0) with tracking + faces (class 1)
        res = yolo.track(frame, persist=True, classes=[0, 1], verbose=False, tracker="bytetrack.yaml")
        if res[0].boxes is None:
            if writer: writer.write(frame)
            continue

        all_boxes = res[0].boxes.xyxy.cpu().numpy()
        all_cls = res[0].boxes.cls.cpu().numpy().astype(int)
        all_ids = res[0].boxes.id.cpu().numpy().astype(int) if res[0].boxes.id is not None else np.full(len(all_boxes), -1, dtype=int)

        person_mask = all_cls == 0
        face_mask = all_cls == 1
        bboxes = all_boxes[person_mask]
        tids = all_ids[person_mask]
        face_boxes = all_boxes[face_mask]

        valid = tids >= 0
        bboxes, tids = bboxes[valid], tids[valid]
        if len(bboxes) == 0:
            if writer: writer.write(frame)
            continue

        # Update positions
        cur_pos = {}
        for bb, tid in zip(bboxes, tids):
            cx, cy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
            history[tid].append((cx, cy))
            if len(history[tid]) > HISTORY_LEN:
                history[tid].pop(0)
            cur_pos[tid] = (np.mean([p[0] for p in history[tid]]),
                            np.mean([p[1] for p in history[tid]]))

        # Age estimation
        if fnum % AGE_INTERVAL == 0:
            for bb, tid in zip(bboxes, tids):
                face_bb = find_face_for_person(bb, face_boxes)
                age = get_age(mivolo, frame, bb, face_bb)
                if age is not None:
                    age_readings[tid].append(age)
                    if len(age_readings[tid]) > AGE_HISTORY_LEN:
                        age_readings[tid].pop(0)
                    # Use median — robust to outlier frames
                    ages[tid] = float(np.median(age_readings[tid]))

        # Height-based child detection: compare each person's bbox height to nearby people
        heights = {tid: bb[3] - bb[1] for tid, bb in zip(tids, bboxes)}
        is_child_by_height = {}
        if len(heights) >= 2:
            max_height = max(heights.values())
            for tid, h in heights.items():
                ratio = h / max_height if max_height > 0 else 1.0
                is_child_by_height[tid] = ratio < CHILD_HEIGHT_RATIO

        # Groups
        if len(cur_pos) >= 2:
            groups = cluster_groups(cur_pos)

        # Run PA-100K child classifier on each person crop
        clf_scores = {}
        if fnum % AGE_INTERVAL == 0:
            for bb, tid in zip(bboxes, tids):
                x1, y1, x2, y2 = [max(0, int(c)) for c in bb]
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    inp = child_tf(crop).unsqueeze(0).half().cuda()
                    with torch.no_grad():
                        logits = child_clf(inp)
                        prob = torch.softmax(logits, dim=1)[0, 1].item()
                    clf_history[tid].append(prob)
                    if len(clf_history[tid]) > 20:
                        clf_history[tid].pop(0)

        # Smoothed classifier score: use 75th percentile over recent history
        # This means if the model sees a child confidently in 25%+ of frames, they stay flagged
        for tid in tids:
            if clf_history[tid]:
                clf_scores[tid] = float(np.percentile(clf_history[tid], 75))
            else:
                clf_scores[tid] = 0

        # Classify: classifier is primary, MiVOLO secondary
        is_child = {}
        for tid in tids:
            clf_prob = clf_scores.get(tid, 0)
            age = ages.get(tid)
            age_child = age is not None and age < CHILD_AGE_THRESHOLD

            is_child[tid] = clf_prob > 0.4 or (clf_prob > 0.15 and age_child)

        # Alerts
        alerts = []
        for tid in tids:
            if is_child.get(tid, False):
                gid = groups.get(tid, -1)
                non_children = [t for t in tids if groups.get(t) == gid
                                and not is_child.get(t, False) and t in cur_pos]
                if non_children:
                    gcx = np.mean([cur_pos[t][0] for t in non_children])
                    gcy = np.mean([cur_pos[t][1] for t in non_children])
                    cp = cur_pos.get(tid, (0, 0))
                    d = np.hypot(cp[0]-gcx, cp[1]-gcy)
                    if d > STRAY_DISTANCE_THRESHOLD:
                        alerts.append((tid, d, cp, (gcx, gcy)))

        # Draw
        palette = [(255,100,100),(100,255,100),(100,100,255),(255,255,100),(255,100,255),(100,255,255)]
        for bb, tid in zip(bboxes, tids):
            x1, y1, x2, y2 = map(int, bb)
            age = ages.get(tid)
            child = is_child.get(tid, False)
            h_ratio = heights.get(tid, 0) / max(heights.values()) if heights else 0
            gid = groups.get(tid, -1)
            col = palette[gid % len(palette)] if gid >= 0 else (180,180,180)
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 3 if child else 2)
            lbl = f"ID:{tid}"
            tag = "CHILD" if child else "ADULT"
            age_str = f"{age:.0f}y" if age is not None else "?"
            lbl += f" {tag} ({age_str} h:{h_ratio:.0%}) G{gid}"
            cv2.putText(frame, lbl, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        for tid, d, cp, gc in alerts:
            cv2.line(frame, (int(cp[0]),int(cp[1])), (int(gc[0]),int(gc[1])), (0,0,255), 3)
            cv2.circle(frame, (int(cp[0]),int(cp[1])), 20, (0,0,255), 3)
            cv2.putText(frame, f"ALERT: CHILD STRAYING ({d:.0f}px)",
                        (int(cp[0])-120, int(cp[1])-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        elapsed = time.time() - t0
        pfps = fnum / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"F{fnum} | {pfps:.1f}FPS | People:{len(tids)} | Alerts:{len(alerts)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if writer: writer.write(frame)
        if args.show:
            cv2.imshow('Child Safety', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"\n{fnum} frames in {elapsed:.1f}s ({pfps:.1f} FPS)")
    if args.output: print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
