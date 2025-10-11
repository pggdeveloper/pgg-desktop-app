from pathlib import Path

BASE_API_URL        = "http://localhost:3001/api/v1/"
LOGIN_ENDPOINT_URL  = "auth/login"
DEBUG_MODE          = True
PHOTO_PREVIEW       = False
BASE_DIR            = Path("./data")   # carpeta base de datasets
CAM_INDEX           = 0                # índice del dispositivo (0,1,2,…)
FPS                 = 30               # objetivo de FPS
PHOTO_COUNT         = 15               # nº de fotos por ojo
VIDEO_SECS          = 60               # duración del vídeo
# Si tu cámara UVC soporta 3840x1080 (1080p por ojo), ponlos aquí:
REQ_WIDTH           = 3840             # 2 * 1920 (SBS)
REQ_HEIGHT          = 1080
REALSENSE_WIDTH    = 1920             # por ojo
REALSENSE_HEIGHT   = 1080             # por ojo
SAVE_JPEG           = False            # True=JPEG (más liviano), False=PNG (sin pérdida)
PHOTO_DELAY         = 3             # segundos entre fotos
SAVE_PHOTOS         = True         # True=guardar fotos, False=no guardar fotos
START_RECORDING_DELAY = 5             # segundos antes de empezar a grabar
