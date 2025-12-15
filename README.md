# Vision-Based ADAS System

## Overview
This project implements a camera-based Advanced Driver Assistance System (ADAS) including lane detection, vehicle detection, distance estimation, and forward collision warning.

## Features
- Lane detection using classical computer vision (OpenCV)
- Vehicle detection using YOLOv8
- Distance estimation using monocular camera geometry
- Time-To-Collision (TTC) computation
- Forward Collision Warning (FCW)

## Tech Stack
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy

## System Architecture
Camera → Lane Detection → Vehicle Detection → Distance Estimation → Collision Warning

## How to Run
```bash
pip install -r requirements.txt
python main.py
