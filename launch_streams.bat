@echo off
title MediaMTX — Traffic Node Streams
echo ============================================================
echo  Capstone Traffic System — RTSP Stream Launcher
echo ============================================================
echo.
echo  Starting MediaMTX RTSP server...
echo  Streams will be available at:
echo    rtsp://localhost:8554/node1  (video1.mp4 - Intersection A)
echo    rtsp://localhost:8554/node2  (video2.mp4 - Intersection B)
echo.
echo  Keep this window open while running cv_pipeline.py
echo  Press Ctrl+C to stop all streams.
echo ============================================================
echo.

cd /d "%~dp0mediamtx"
mediamtx.exe mediamtx.yml
