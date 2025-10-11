#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graba fotos y vídeo desde ZED 2i (modo UVC, sin GPU) separando LEFT/RIGHT.
Requiere:  python -m pip install opencv-python
"""

import cv2
import os
import csv
import time
from pathlib import Path
from config import DEBUG_MODE, BASE_DIR, CAM_INDEX, FPS, PHOTO_COUNT, VIDEO_SECS, REQ_WIDTH, REQ_HEIGHT, SAVE_JPEG, PHOTO_DELAY, SAVE_PHOTOS

def try_set(vc, prop, value):
    """Intenta setear una propiedad y retorna el valor leído."""
    vc.set(prop, value)
    return vc.get(prop)

def disable_auto(vc):
    # Muchos drivers ignoran estos flags, pero intentamos:
    # Auto exposición (0.25: manual en algunos drivers DirectShow)
    vc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # WB manual
    vc.set(cv2.CAP_PROP_AUTO_WB, 0)
    # (Opcional) valores fijos si tu entorno lo requiere:
    # vc.set(cv2.CAP_PROP_EXPOSURE, -6)       # rango depende del driver
    # vc.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

def split_sbs(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    left  = frame[:, :mid]
    right = frame[:, mid:]
    return left, right
    
def print_message(msg: str):
    if DEBUG_MODE:
        print(msg)

def create_folders(caravana: int):
    carpeta = BASE_DIR / str(caravana)
    d_left  = carpeta / "left"
    d_right = carpeta / "right"
    d_left.mkdir(parents=True, exist_ok=True)
    d_right.mkdir(parents=True, exist_ok=True)
    print_message(f"Guardando datos en: {carpeta}")
    return carpeta, d_left, d_right

def test_camera():
    """Testea si la cámara está accesible y muestra sus propiedades."""
    print_message(f"Abriendo cámara index={CAM_INDEX}…")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)  # intenta DirectShow (Windows)
    if not cap.isOpened():
        print_message(f"[ERROR] No se pudo abrir la cámara (indice {CAM_INDEX}).")
        return None, None, None

    print_message("Cámara abierta. Propiedades:")
    w = try_set(cap, cv2.CAP_PROP_FRAME_WIDTH, REQ_WIDTH)
    h = try_set(cap, cv2.CAP_PROP_FRAME_HEIGHT, REQ_HEIGHT)
    fps = try_set(cap, cv2.CAP_PROP_FPS, FPS)
    print_message(f" - Resolución: {w}x{h}")
    print_message(f" - FPS:        {fps}")
    disable_auto(cap)
    ok, test = cap.read()
    if not ok or test is None or getattr(test, "size", 0) == 0:
        return None, None, None
    H, W = test.shape[:2]

    return cap, H, W

def save_frames(caravana, d_left, d_right, ext, left, right, idx, time_now, previous_time):
    if (time_now - previous_time) < PHOTO_DELAY or not SAVE_PHOTOS:
        return
    fL = d_left  / f"{caravana}-L-{idx:02d}{ext}"
    (cv2.imwrite(str(fL), left) and print_message(f"[OK] {fL}"))

    if right is not None:
        fR = d_right / f"{caravana}-R-{idx:02d}{ext}"
        (cv2.imwrite(str(fR), right) and print_message(f"[OK] {fR}"))

def main():
    caravana =  30000 #input("Ingrese número de caravana: ").strip()
    if not caravana:
        print_message("Número de caravana vacío. Saliendo.")
        return

    carpeta, d_left, d_right = create_folders(caravana)
    
    
    cap, H, W = test_camera()
    if cap is None or H is None or W is None:
        print_message("[ERROR] No se pudo iniciar la cámara. Saliendo.")
        return
    

    # Heurística: si el ancho es ~2× el alto o si al partir hay dos mitades válidas → SBS
    is_sbs = (W >= 2 * (H // 2))  # condición laxa
    if not is_sbs:
        print_message(f"[WARN] No parece SBS (W={W}, H={H}). Continuo guardando frame completo como LEFT.")
    else:
        print_message(f"[INFO] Detectado SBS: frame {W}x{H} → por ojo ~{W//2}x{H}")

    ext = ".jpg" if SAVE_JPEG else ".png"

    # --- 1) Captura de fotos ---
    
    # for i in range(1, PHOTO_COUNT + 1):
    #     ret, frame = cap.read()
    #     if not ret:
    #         print_message(f"[WARN] Foto {i}: frame no capturado, continúo…")
    #         continue

    #     if is_sbs:
    #         left, right = split_sbs(frame)
    #     else:
    #         left, right = frame, None

    #     fL = d_left  / f"{caravana}-L-{i:02d}{ext}"
    #     (cv2.imwrite(str(fL), left) and print_message(f"[OK] {fL}"))

    #     if right is not None:
    #         fR = d_right / f"{caravana}-R-{i:02d}{ext}"
    #         (cv2.imwrite(str(fR), right) and print_message(f"[OK] {fR}"))

    #     time.sleep(0.15)  # pequeña pausa

    # --- 2) Grabación de vídeo por ojo ---
    # Elegimos MJPG en AVI (intra-frame: mejor para análisis que H.264 con compresión temporal).
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    w_eye  = (W // 2) if is_sbs else W
    h_eye  = H

    vL_path = carpeta / f"{caravana}-L.avi"
    vR_path = carpeta / f"{caravana}-R.avi"

    outL = cv2.VideoWriter(str(vL_path), fourcc, FPS, (w_eye, h_eye))
    outR = cv2.VideoWriter(str(vR_path), fourcc, FPS, (w_eye, h_eye)) if is_sbs else None

    # Timestamps
    ts_csv = carpeta / "timestamps.csv"
    with open(ts_csv, "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["frame_idx", "t_ns", "eye"])  # eye: L o R

        print_message(f"Iniciando grabación de vídeo ({VIDEO_SECS}s)…")
        previous_time = time.time()
        actual_time = previous_time
        idx = 0
        while actual_time - previous_time < VIDEO_SECS:
            ret, frame = cap.read()
            if not ret or frame is None or getattr(frame, "size", 0) == 0:
                print_message("[WARN] Fotograma perdido durante la grabación")
                continue

            actual_time_nano_seconds = time.time_ns()
            if is_sbs:
                left, right = split_sbs(frame)
                outL.write(left); wr.writerow([idx, actual_time_nano_seconds, "L"])
                outR.write(right); wr.writerow([idx, actual_time_nano_seconds, "R"])
            else:
                left, right = frame, None
                outL.write(frame); wr.writerow([idx, actual_time_nano_seconds, "L"])

            save_frames(caravana, d_left, d_right, ext, left, right, idx, actual_time, previous_time)

            idx += 1
            actual_time = time.time()
    caravana += 1

    outL.release()
    if outR is not None: outR.release()
    cap.release()

    print_message(f"[OK] Vídeo LEFT:  {vL_path}")
    if is_sbs:
        print_message(f"[OK] Vídeo RIGHT: {vR_path}")
    print_message(f"[OK] Timestamps:  {ts_csv}")

if __name__ == "__main__":
    main()
