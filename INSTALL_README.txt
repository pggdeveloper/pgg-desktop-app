================================================================================
PGG Cattle Monitoring Application - Installation Guide
================================================================================

Thank you for choosing PGG Cattle Monitor!

This installation guide will help you set up the application on your Windows
computer for livestock monitoring using Intel RealSense D455i and Stereolabs
ZED 2i cameras.

================================================================================
SYSTEM REQUIREMENTS
================================================================================

Minimum Requirements:
- Windows 10 64-bit or later
- 8 GB RAM
- Intel Core i5 processor or equivalent
- USB 3.0 ports for camera connectivity
- 50 GB free disk space (for recordings)

Recommended Requirements:
- Windows 11 64-bit
- 16 GB RAM
- Intel Core i7 processor or better
- USB 3.1 Gen 2 ports
- SSD storage for improved recording performance

================================================================================
REQUIRED DEPENDENCIES
================================================================================

Before installing PGG Cattle Monitor, you MUST install the following camera
SDKs on your system:

1. Intel RealSense SDK 2.0 or later
   - Download from: https://github.com/IntelRealSense/librealsense/releases
   - Choose the latest Windows installer (.exe)
   - Follow the installation wizard
   - Restart your computer after installation

2. Stereolabs ZED SDK 4.0 or later (if using ZED 2i cameras)
   - Download from: https://www.stereolabs.com/developers/release/
   - Choose the latest ZED SDK for Windows
   - Follow the installation wizard
   - CUDA is optional for this application (CPU-only mode supported)
   - Restart your computer after installation

3. Microsoft Visual C++ Redistributables
   - Usually installed automatically by the camera SDKs
   - If needed, download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

================================================================================
INSTALLATION STEPS
================================================================================

1. Install the required dependencies listed above (if not already installed)

2. Run the PGG Cattle Monitor installer (PGG_Cattle_Monitor_Setup.exe)

3. Follow the installation wizard:
   - Accept the license agreement
   - Choose installation directory (default: C:\Program Files\PGG Cattle Monitor)
   - Select components to install
   - Choose Start Menu folder
   - Create desktop shortcut (optional)

4. Click "Install" to begin installation

5. Wait for the installation to complete

6. Click "Finish" to exit the installer

================================================================================
FIRST RUN CONFIGURATION
================================================================================

When you first launch PGG Cattle Monitor:

1. Connect your cameras:
   - Intel RealSense D455i: Connect to USB 3.0 port
   - ZED 2i cameras: Connect each to separate USB 3.0 ports

2. Launch the application from:
   - Desktop shortcut, or
   - Start Menu > PGG Cattle Monitor

3. The application will automatically detect connected cameras

4. (Optional) Perform multi-camera calibration:
   - Use the calibration tool to establish spatial relationships
   - Follow the on-screen instructions
   - Save calibration data for future sessions

================================================================================
CAMERA SETUP GUIDELINES
================================================================================

For optimal performance:

- Use high-quality USB 3.0 cables (maximum 3 meters length)
- Connect cameras directly to motherboard USB ports (avoid hubs if possible)
- Ensure adequate lighting for best image quality
- Mount cameras securely with clear view of the monitoring area
- For multi-camera setups, ensure fields of view overlap for 3D reconstruction

Recommended camera placement:
- Height: 2-3 meters above ground
- Angle: 30-45 degrees downward
- Distance: 2-5 meters from monitoring area
- Overlap: 30-50% field of view overlap between cameras

================================================================================
FEATURES
================================================================================

PGG Cattle Monitor includes:

✓ Multi-camera synchronized recording (RealSense + ZED 2i)
✓ Real-time 3D point cloud generation
✓ Cattle volume and weight estimation
✓ Motion analysis and activity monitoring
✓ Trajectory tracking and path analysis
✓ Scene understanding (floor, walls, objects detection)
✓ Body condition scoring (BCS)
✓ Health monitoring (coat, lesions, anemia, posture)
✓ Environmental assessment (cleanliness, water, feed)
✓ IMU data integration (for RealSense D455i)
✓ Comprehensive CSV data export

================================================================================
GETTING STARTED
================================================================================

1. Launch PGG Cattle Monitor

2. Verify cameras are detected in the main window

3. Configure recording settings:
   - Recording duration
   - Frame rate (15, 30, 60, or 90 fps)
   - Output directory
   - Enable desired analytics features

4. (Optional) Load calibration data if performing multi-camera recording

5. Click "Start Recording" to begin monitoring

6. Recordings and analytics will be saved to the configured output directory

================================================================================
OUTPUT FILES
================================================================================

After recording, you will find the following files in the output directory:

Video Files:
- *-rgb.mp4: RGB video stream
- *-depth.mp4: Depth map visualization
- *-ir_left.mp4: Left infrared stream
- *-ir_right.mp4: Right infrared stream

Point Clouds:
- *-pointcloud_*.ply: 3D point clouds (can be viewed in CloudCompare, MeshLab)

Analytics (CSV format):
- *-volume_measurements.csv: Volume and weight estimates
- *-motion_analysis.csv: Motion, speed, and gait data
- *-trajectory_points.csv: 3D trajectory data
- *-scene_report.csv: Scene understanding data
- *-imu_report.csv: IMU sensor fusion data
- *-bcs_*.csv: Body condition scoring results
- *-health_*.csv: Health monitoring results

Metadata:
- *-metadata.json: Recording session metadata
- *-timestamps.csv: Frame timestamp log

================================================================================
TROUBLESHOOTING
================================================================================

Problem: Cameras not detected
Solution:
  - Check USB connections
  - Verify camera SDKs are installed
  - Restart the application
  - Try different USB ports

Problem: Slow performance
Solution:
  - Reduce frame rate
  - Disable some analytics features
  - Close other applications
  - Ensure adequate CPU/RAM available

Problem: Recording fails to start
Solution:
  - Check available disk space
  - Verify output directory is writable
  - Check camera permissions in Windows settings

Problem: Poor depth quality
Solution:
  - Improve lighting conditions
  - Clean camera lenses
  - Adjust camera distance from subject
  - Enable depth filtering in settings

================================================================================
SUPPORT
================================================================================

For technical support, please contact:
- Email: support@pgg.com
- Website: https://www.pgg.com/support
- Documentation: See docs/ folder in installation directory

For camera SDK support:
- Intel RealSense: https://github.com/IntelRealSense/librealsense/issues
- ZED SDK: https://support.stereolabs.com/

================================================================================
LICENSE
================================================================================

PGG Cattle Monitor is proprietary software.
Copyright (C) 2025 PGG. All rights reserved.

See LICENSE.txt for full license terms.

================================================================================
THANK YOU
================================================================================

Thank you for choosing PGG Cattle Monitor for your livestock monitoring needs.
We hope this application helps you improve the health and welfare of your
cattle through advanced computer vision technology.

Happy monitoring!

================================================================================
