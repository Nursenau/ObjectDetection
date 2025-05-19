import cv2
import os
import shutil
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# Dosya yolları
MODEL_PATH = "C:/Users/ASUS/PycharmProjects/HelmetVestDetection/HelmetVestDetection/train4/weights/best.pt"
VIDEO_PATH = "video_predictions/video_4.avi"
LOG_PATH = "uyari_log.txt"
DETECTIONS_FOLDER = "detections"

# Her çalıştırmada detections klasörünü temizle ve oluştur
if os.path.exists(DETECTIONS_FOLDER):
    shutil.rmtree(DETECTIONS_FOLDER)
os.makedirs(DETECTIONS_FOLDER)

# Uyarı eşiği (saniye)
UYARI_ESIK_SANIYE = 2

# Model ve tracker başlat
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30)

# Video aç
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    raise ValueError("FPS değeri 0! Video yüklenemedi.")
frame_threshold = int(fps * UYARI_ESIK_SANIYE)

# Takip geçmişi ve uyarılar
etiket_gecmisi = defaultdict(lambda: {"no-helmet": 0, "no-vest": 0})
uyarilanlar = {"no-helmet": set(), "no-vest": set()}
uyari_var = False

# Log dosyasını sıfırla
with open(LOG_PATH, "w", encoding="utf-8") as log_file:
    log_file.write("Uyarı raporu başlatıldı\n")

# Video kareleri işleniyor
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)[0]
    detections = []

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r
        label = results.names[int(cls)]
        if label in ["helmet", "no-helmet", "vest", "no-vest"]:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        label = track.get_det_class()

        if label in ["no-helmet", "no-vest"]:
            etiket_gecmisi[track_id][label] += 1

            if etiket_gecmisi[track_id][label] == frame_threshold and track_id not in uyarilanlar[label]:
                uyarilanlar[label].add(track_id)
                uyari_var = True

                uyarı_metni = f"UYARI! İşçi {track_id}: Uzun süre {label.replace('-', ' ')} çalışıyor!"
                print(uyarı_metni)

                # Log dosyasına yaz
                with open(LOG_PATH, "a", encoding="utf-8") as log_file:
                    log_file.write(uyarı_metni + "\n")

                # İlgili işçinin görüntüsünü kaydet
                # Takip kutusunu al
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    image_path = os.path.join(DETECTIONS_FOLDER, f"isçi_{track_id}_{label}.jpg")
                    cv2.imwrite(image_path, person_crop)

cap.release()

# Eğer hiç uyarı yoksa:
if not uyari_var:
    mesaj = "Tüm işçiler kurallara uyuyor."
    print(mesaj)
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(mesaj + "\n")
