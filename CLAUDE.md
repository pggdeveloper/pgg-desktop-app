# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyQt5 desktop application for capturing photos from external USB cameras. The application features a modern dark UI theme and is designed to work cross-platform (Windows, macOS, Linux).

## Setup and Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run the main application
python app.py
```

## Architecture

### Application Structure

The application follows a screen-based navigation pattern:

- **app.py**: Entry point that initializes PyQt5, applies global theme, and creates the MainWindow
- **screens/main_screen.py**: MainWindow orchestrates screen transitions using QStackedWidget with crossfade animations. Currently displays CaptureScreen directly (LoginScreen is commented out but available)
- **screens/capture_screen.py**: Camera capture interface that uses utils.py to enumerate and open USB cameras
- **screens/login_screen.py**: Authentication screen (currently disabled) that makes POST requests to the API endpoint defined in config.py

### Key Components

- **components/animated_button.py**: Custom QPushButton with opacity/shadow animations (animation code currently commented out)
- **components/password_field.py**: QLineEdit with toggle visibility button for password input

### Configuration and Constants

- **config.py**: API endpoints and debug mode flag
  - `BASE_API_URL`: Backend API base URL (default: http://localhost:3001/api/v1/)
  - `LOGIN_ENDPOINT_URL`: Login endpoint path
  - `DEBUG_MODE`: Enables preview display and console logging in CaptureScreen

- **constants.py**: Application-wide theme system with dark color palette
  - `APP`: Window dimensions, title, border radius
  - `THEME`: Complete design system including colors, sizes, animations, fonts, and QSS stylesheets
  - All UI styling is centralized here; colors use hex values with semantic names (bg, primary, text_muted, etc.)

### Camera Handling (utils.py)

The `utils.py` module provides cross-platform USB camera detection and filtering:

- **enumerate_usb_external_cameras()**: Returns list of external USB cameras, filtering out virtual cameras (OBS, ManyCam, etc.) and internal/built-in cameras (FaceTime, integrated webcams)
- **Platform-specific implementations**:
  - Windows: Uses PowerShell PnP Device enumeration + DirectShow backend
  - macOS: Uses AVFoundation API (via PyObjC) + system_profiler for USB detection
  - Linux: Uses pyudev or /dev/v4l/by-id symlinks + V4L2 backend
- **open_camera_hint()**: Opens camera using the appropriate backend based on the hint (index or device path) returned by enumerate function

Camera detection flow:
1. Probe OS for video devices using native APIs
2. Filter by USB bus connection
3. Exclude virtual/internal cameras by name patterns
4. Verify cameras can be opened and read frames
5. Return list with backend-specific open hints

### Session Management

- MainWindow stores `session_data` dict after successful login (currently unused as login is disabled)
- CaptureScreen can receive session data via `set_session_data()` slot for future API integration

## Important Patterns

### Theme Application

The global QSS stylesheet is applied once in app.py via `app.setStyleSheet(THEME["qss"]["base"])`. Individual widgets use objectName for specific styling (e.g., `objectName="Card"`, `objectName="H1"`).

### Screen Transitions

MainWindow uses `_crossfade_to(widget)` for animated transitions between screens with fade-out/fade-in effect using QPropertyAnimation on QGraphicsOpacityEffect.

### Camera Capture Flow

1. Click "Tomar foto" button
2. Enumerate USB external cameras using platform-specific detection
3. Open first available camera (or fallback to index 0)
4. Capture single frame
5. Release camera immediately
6. Save as JPEG with timestamp to `./snaps/` directory
7. Show preview (if DEBUG_MODE enabled) and success feedback

### Platform Dependencies

Platform-specific packages are conditionally installed via requirements.txt:
- Linux: pyudev for device enumeration
- macOS: pyobjc for AVFoundation camera access
