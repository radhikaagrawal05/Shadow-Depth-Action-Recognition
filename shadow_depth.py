import cv2
import mediapipe as mp
import numpy as np

# ================= PHYSICS CONSTANTS =================
K = 1500.0                  # scaling constant
TOUCH_THRESHOLD = 20.0       # cm threshold (tuned for demo)

# ================= MEDIAPIPE INIT =================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.6)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("Physics Model: Z = K / sqrt(Shadow_Area)")
print("Move your hand near face to change shadow")
print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    hand_results = hands_detector.process(rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                hx = int(lm.x * frame.shape[1])
                hy = int(lm.y * frame.shape[0])
                cv2.circle(frame, (hx, hy), 4, (255, 0, 0), -1)

        cv2.putText(frame, "Hand Detected", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, "Hand influencing shadow", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            face_roi = frame[y1:y1+bh, x1:x1+bw]
            if face_roi.size == 0:
                continue

            # ================= SHADOW EXTRACTION =================
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            _, shadow_mask = cv2.threshold(
                gray, 90, 255, cv2.THRESH_BINARY_INV
            )

            cv2.imshow("Shadow Mask", shadow_mask)

            # ================= SHADOW AREA =================
            shadow_area = np.count_nonzero(shadow_mask)

            # ================= PHYSICS DEPTH MODEL =================
            Z = K / np.sqrt(shadow_area + 1)

            # ================= ACTION CLASSIFICATION =================
            if Z < TOUCH_THRESHOLD:
                label = "Touching Face"
                color = (0, 0, 255)
            else:
                label = "Not Touching"
                color = (0, 255, 0)

            # ================= INTENSITY LOSS MATRIX =================
            intensity_loss = 255 - gray

            norm_matrix = cv2.normalize(
                intensity_loss, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            heatmap = cv2.applyColorMap(norm_matrix, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (bw, bh))

            # ================= OVERLAY =================
            overlay = cv2.addWeighted(face_roi, 0.3, heatmap, 0.7, 0)
            frame[y1:y1+bh, x1:x1+bw] = overlay

            cv2.rectangle(frame, (x1, y1), (x1+bw, y1+bh), color, 2)

            cv2.putText(
                frame,
                f"Distance Z: {Z:.2f} cm",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            print("Shadow Area:", shadow_area, "| Z:", round(Z, 2), "cm")

    cv2.imshow("Physics-Based Shadow Depth Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
