
from PyQt5 import QtCore

APP = {
    "TITLE": "Cámara",
    "WIDTH": 960,
    "HEIGHT": 640,
    "WINDOW_ROUND_RADIUS": 12,
}

THEME = {
    "colors": {
        "bg": "#0f172a",              # azul muy oscuro (slate-900)
        "bg_card": "#111827",         # gris/azul oscuro
        "primary": "#2563eb",         # azul principal
        "primary_hover": "#1d4ed8",
        "primary_pressed": "#1e40af",
        "text": "#e5e7eb",            # claro
        "text_muted": "#9ca3af",      # gris medio
        "input_bg": "#0b1220",
        "input_border": "#273449",
        "input_border_focus": "#3b82f6",
        "danger": "#ef4444",
        "success": "#10b981",
        "shadow": "#000000",
    },
    "sizes": {
        "font_base": 14,
        "font_h1": 26,
        "font_button": 16,
        "radius_sm": 8,
        "radius_md": 12,
        "btn_height": 46,
        "input_height": 42,
        "input_padding_h": 12,
        "content_maxw": 440,
    },
    "anim": {
        "hover_ms": 180,
        "press_ms": 120,
        "ease": QtCore.QEasingCurve.OutCubic,
        "shadow_hover_blur": 24.0,
        "shadow_hover_y": 12.0,
        "shadow_rest_blur": 12.0,
        "shadow_rest_y": 6.0,
        "opacity_hover": 1.0,
        "opacity_rest": 0.94,
        "fade_ms": 240,
    },
    "fonts": {
        "family": "Inter, Segoe UI, Helvetica, Arial",
    },
    # Hoja de estilo global (QSS) ensamblada con la paleta/tamaños
    "qss": {}
}

THEME["qss"]["base"] = f"""
* {{
    font-family: {THEME['fonts']['family']};
    color: {THEME['colors']['text']};
    font-size: {THEME['sizes']['font_base']}px;
}}
QMainWindow, QWidget {{
    background-color: {THEME['colors']['bg']};
}}
/* Tarjetas/containers */
.QFrame#Card {{
    background-color: {THEME['colors']['bg_card']};
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: {THEME['sizes']['radius_md']}px;
}}
/* Títulos */
QLabel#H1 {{
    font-size: {THEME['sizes']['font_h1']}px;
    font-weight: 700;
    color: {THEME['colors']['text']};
}}
QLabel#Muted {{
    color: {THEME['colors']['text_muted']};
}}
/* Inputs */
QLineEdit {{
    background-color: {THEME['colors']['input_bg']};
    border: 1px solid {THEME['colors']['input_border']};
    border-radius: {THEME['sizes']['radius_sm']}px;
    padding-left: {THEME['sizes']['input_padding_h']}px;
    padding-right: {THEME['sizes']['input_padding_h']}px;
    height: {THEME['sizes']['input_height']}px;
    selection-color: {THEME['colors']['bg']};
    selection-background-color: {THEME['colors']['text']};
}}
QLineEdit:focus {{
    border: 1px solid {THEME['colors']['input_border_focus']};
}}
/* Botones principales */
QPushButton#PrimaryButton {{
    background-color: {THEME['colors']['primary']};
    border: none;
    border-radius: {THEME['sizes']['radius_sm']}px;
    height: {THEME['sizes']['btn_height']}px;
    font-size: {THEME['sizes']['font_button']}px;
    font-weight: 600;
    padding: 0 16px;
}}
QPushButton#PrimaryButton:hover {{
    background-color: {THEME['colors']['primary_hover']};
}}
QPushButton#PrimaryButton:pressed {{
    background-color: {THEME['colors']['primary_pressed']};
}}
QPushButton#PrimaryButton:disabled {{
    background-color: #1f2a44;
    color: #6b7280;
}}
/* Enfatizar foco accesible en botones (keyboard focus) */
QPushButton#PrimaryButton:focus {{
    outline: none;
    border: 2px solid {THEME['colors']['input_border_focus']};
}}
/* Botón ícono (ojito) embebido en el input */
QToolButton#IconInline {{
    background: transparent;
    border: none;
    padding: 0 6px;
    color: {THEME['colors']['text_muted']};
}}
QToolButton#IconInline:hover {{
    color: {THEME['colors']['text']};
}}
/* Etiquetas de feedback */
QLabel#Error {{
    color: {THEME['colors']['danger']};
    font-weight: 600;
}}
QLabel#Success {{
    color: {THEME['colors']['success']};
    font-weight: 600;
}}
"""
