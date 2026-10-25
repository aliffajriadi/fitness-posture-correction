@echo off
title Fitness Posture Correction - Video Demonstration
:MENU
cls
echo ========================================================
echo   FITNESS POSTURE CORRECTION - VIDEO DEMO LAUNCHER
echo   Analisis Perbandingan YOLOv12 dan MediaPipe-MLP
echo ========================================================
echo.
echo Pilih video sampel yang ingin didemokan ke dosen penguji:
echo.
echo   [1] Push-up POSTUR BENAR (Good Form)
echo   [2] Push-up POSTUR SALAH (Bad Form - Pinggul Turun)
echo   [3] Squat POSTUR BENAR   (Good Form)
echo   [4] Squat POSTUR SALAH   (Bad Form - Badan Bungkuk)
echo   [5] Keluar
echo.
set /p pilihan="Masukkan nomor pilihan [1-5]: "

if "%pilihan%"=="1" goto PUSHUP_GOOD
if "%pilihan%"=="2" goto PUSHUP_BAD
if "%pilihan%"=="3" goto SQUAT_GOOD
if "%pilihan%"=="4" goto SQUAT_BAD
if "%pilihan%"=="5" goto EXIT
goto MENU

:PUSHUP_GOOD
echo.
echo Menjalankan Demo: Push-up Good Form (Tekan 1/2 untuk switch model, q untuk kembali)
.venv\Scripts\python.exe -m tools.demo --video "data/sources/Dataset Subject-wise/subject_001/push_up/good/side/subject_001_push_up_good_side.mp4" --pipeline mlp
goto MENU

:PUSHUP_BAD
echo.
echo Menjalankan Demo: Push-up Bad Form (Koreksi postur pinggul)
.venv\Scripts\python.exe -m tools.demo --video "data/sources/Dataset Subject-wise/subject_001/push_up/bad/side/subject_001_push_up_bad_side.mp4" --pipeline mlp
goto MENU

:SQUAT_GOOD
echo.
echo Menjalankan Demo: Squat Good Form
.venv\Scripts\python.exe -m tools.demo --video "data/sources/Dataset Subject-wise/subject_001/squat/good/side/subject_001_squat_good_side.mp4" --pipeline mlp
goto MENU

:SQUAT_BAD
echo.
echo Menjalankan Demo: Squat Bad Form (Koreksi postur membungkuk)
.venv\Scripts\python.exe -m tools.demo --video "data/sources/Dataset Subject-wise/subject_001/squat/bad/side/subject_001_squat_bad_side.mp4" --pipeline mlp
goto MENU

:EXIT
echo.
echo Keluar dari demo.
exit /b
