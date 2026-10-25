@echo off
title Fitness Posture Correction - Live Webcam Demo
echo ========================================================
echo   FITNESS POSTURE CORRECTION - LIVE WEBCAM DEMO
echo   Analisis Perbandingan YOLOv12 dan MediaPipe-MLP
echo ========================================================
echo.
echo Petunjuk Kontrol Keyboard saat demo terbuka:
echo   - Tekan [1] : Beralih ke Pipeline MediaPipe + MLP
echo   - Tekan [2] : Beralih ke Pipeline YOLOv12
echo   - Tekan [q] : Keluar dari demo
echo.
echo Membuka kamera webcam (Kamera 0)...
echo Pastikan tubuh Anda tampak dari samping (side-view).
echo.

.venv\Scripts\python.exe -m tools.demo --camera 0 --pipeline mlp

echo.
echo Sesi demo selesai.
pause
