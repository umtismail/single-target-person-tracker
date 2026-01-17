import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------ AYARLAR ------------------
SIM_THRESHOLD = 0.90
ZOOM_SCALE = 2.0
REF_EMB_COUNT = 5

# ------------------ GLOBAL ------------------
selected_id = None
selected_embeddings = []
current_boxes = []
current_ids = []
frame_global = None

# ------------------ RE-ID MODEL ------------------
reid_model = models.resnet50(pretrained=True)
reid_model.fc = torch.nn.Identity()
reid_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def draw_target_marker(frame, box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = y1 - 15  # kafanın biraz üstü

    # dış daire
    cv2.circle(frame, (cx, cy), 12, (0, 0, 255), 2)

    # iç nokta
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # artı işareti
    cv2.line(frame, (cx - 8, cy), (cx + 8, cy), (0, 0, 255), 2)
    cv2.line(frame, (cx, cy - 8), (cx, cy + 8), (0, 0, 255), 2)


def extract_embedding(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = reid_model(img)
    return emb.numpy()

def verify_person(candidate_emb):
    scores = [
        cosine_similarity(candidate_emb, ref)[0][0]
        for ref in selected_embeddings
    ]
    return max(scores) > SIM_THRESHOLD

# ------------------ DIGITAL ZOOM ------------------
def digital_zoom(frame, box, scale):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)//2, (y1+y2)//2
    nw, nh = int(w/scale), int(h/scale)

    x1n = max(0, cx - nw//2)
    y1n = max(0, cy - nh//2)
    x2n = min(w, cx + nw//2)
    y2n = min(h, cy + nh//2)

    crop = frame[y1n:y2n, x1n:x2n]
    return cv2.resize(crop, (w, h))

# ------------------ MOUSE ------------------
def mouse_callback(event, x, y, flags, param):
    global selected_id, selected_embeddings

    if event == cv2.EVENT_LBUTTONDOWN and selected_id is None:
        for box, id_ in zip(current_boxes, current_ids):
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                crop = frame_global[y1:y2, x1:x2]
                if crop.size == 0:
                    return

                selected_id = int(id_)
                selected_embeddings.clear()

                for _ in range(REF_EMB_COUNT):
                    selected_embeddings.append(extract_embedding(crop))

                print(f"[LOCK] Hedef seçildi ID {selected_id}")
                break

# ------------------ VIDEO SEÇ ------------------
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Video Seç",
    filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")]
)
if not video_path:
    exit()

# ------------------ MODEL ------------------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("VisionFlow", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("VisionFlow", mouse_callback)

# ------------------ MAIN LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_global = frame.copy()
    display = frame.copy()

    current_boxes.clear()
    current_ids.clear()

    target_visible = False
    target_box = None
    best_candidate = None
    best_score = 0

    results = model.track(frame, persist=True, classes=[0], conf=0.4)

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, id_ in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            current_boxes.append((x1,y1,x2,y2))
            current_ids.append(int(id_))

            if selected_id is None:
                cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(display,f"ID {int(id_)}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            elif int(id_) == selected_id:
                target_visible = True
                target_box = (x1,y1,x2,y2)

            else:
                emb = extract_embedding(crop)
                score = max(
                    cosine_similarity(emb, ref)[0][0]
                    for ref in selected_embeddings
                )
                if score > best_score:
                    best_score = score
                    best_candidate = (int(id_), (x1,y1,x2,y2))

    # ---------- RE-ID ----------
    if selected_id and not target_visible:
        if best_candidate and best_score > SIM_THRESHOLD:
            selected_id = best_candidate[0]
            target_box = best_candidate[1]
            print("[VERIFY] Hedef geri bulundu")

        # ---------- ZOOM ----------
    if target_box:
        # zoom
        display = digital_zoom(display, target_box, ZOOM_SCALE)

        # hedef işareti
        draw_target_marker(display, target_box)

        cv2.putText(
            display,
            "TARGET LOCKED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )




    cv2.imshow("VisionFlow", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("r"):
        selected_id = None
        selected_embeddings.clear()
        print("[RESET]")

cap.release()
cv2.destroyAllWindows()
