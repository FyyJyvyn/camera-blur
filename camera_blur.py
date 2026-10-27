import cv2
import mediapipe as mp
import pygame
import time
import os

# =========================
# LAGU
# =========================

MUSIC_FILE = "lagu.mp3"

# =========================
# LIRIK
# =========================
# Angka = detik sejak lagu mulai

lyrics = [
    (0.0, "Foto kita blur"),
    (3.5, "Tak banyak yang kulihat"),
    (7.0, "Foto kita blur"),
    (10.5, "Tapi banyak yang aku ingat"),
    (14.0, "Foto kita blur"),
    (17.5, "Kita pernah begitu dekat"),
    (21.0, "Foto kita blur"),
    (24.5, "Kini memorinya kudekap"),
]

# =========================
# AUDIO
# =========================

pygame.mixer.init()

if not os.path.exists(MUSIC_FILE):
    print("lagu.mp3 tidak ditemukan!")
    exit()

pygame.mixer.music.load(MUSIC_FILE)

print("Lagu berhasil dimuat.")

# =========================
# KAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak dapat dibuka.")
    pygame.quit()
    exit()

# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# STATUS
# =========================

music_started = False
music_start_time = 0

# =========================
# DETEKSI 2 JARI
# =========================

def detect_two_fingers(hand_landmarks):

    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y

    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y

    return (
        index_up
        and middle_up
        and ring_down
        and pinky_down
    )

# =========================
# CARI LIRIK
# =========================

def get_current_lyric(elapsed):

    current_index = -1

    for i in range(len(lyrics)):

        timestamp = lyrics[i][0]

        if elapsed >= timestamp:
            current_index = i
        else:
            break

    return current_index

# =========================
# TAMPILKAN LIRIK
# =========================

def draw_lyric(frame, text, fade):

    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.95
    thickness = 2

    size = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )[0]

    text_width = size[0]

    x = (w - text_width) // 2
    y = h - 90

    # Gerakan sedikit dari bawah
    y += int(10 * (1 - fade))

    # =====================
    # SHADOW
    # =====================

    cv2.putText(
        frame,
        text,
        (x + 3, y + 3),
        font,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA
    )

    # =====================
    # TEKS
    # =====================

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # =====================
    # GARIS
    # =====================

    line_width = int(text_width * 0.3 * fade)

    if line_width > 0:

        line_x = (w - line_width) // 2
        line_y = y + 18

        cv2.line(
            frame,
            (line_x, line_y),
            (line_x + line_width, line_y),
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

# =========================
# LOOP KAMERA
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # =====================
    # DETEKSI TANGAN
    # =====================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    two_fingers = False

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            if detect_two_fingers(hand_landmarks):

                two_fingers = True

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # =====================
    # MULAI LAGU
    # =====================

    if two_fingers and not music_started:

        pygame.mixer.music.play()

        music_started = True

        music_start_time = time.time()

        print("LAGU DIMULAI")

    # =====================
    # BLUR
    # =====================

    if two_fingers:

        frame = cv2.GaussianBlur(
            frame,
            (51, 51),
            0
        )

    # =====================
    # LIRIK
    # =====================

    if music_started:

        elapsed = (
            time.time()
            - music_start_time
        )

        lyric_index = get_current_lyric(
            elapsed
        )

        if lyric_index >= 0:

            text = lyrics[lyric_index][1]

            start_time = lyrics[lyric_index][0]

            # Waktu sampai lirik berikutnya
            if lyric_index + 1 < len(lyrics):

                next_time = lyrics[
                    lyric_index + 1
                ][0]

            else:

                next_time = start_time + 4

            duration = next_time - start_time

            lyric_elapsed = elapsed - start_time

            # =================
            # FADE
            # =================

            fade = 1.0

            if lyric_elapsed < 0.5:

                fade = lyric_elapsed / 0.5

            elif lyric_elapsed > duration - 0.5:

                fade = (
                    duration - lyric_elapsed
                ) / 0.5

            fade = max(
                0,
                min(1, fade)
            )

            draw_lyric(
                frame,
                text,
                fade
            )

    # =====================
    # STATUS
    # =====================

    if two_fingers:

        status = "2 JARI - BLUR"

    else:

        status = "CLEAR"

    cv2.putText(
        frame,
        status,
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # =====================
    # TAMPILKAN
    # =====================

    cv2.imshow(
        "Camera Trend",
        frame
    )

    # Q = keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# SELESAI
# =========================

cap.release()
hands.close()

pygame.mixer.music.stop()
pygame.quit()

cv2.destroyAllWindows()