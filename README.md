# gendered-trailers-analysis

Computational analysis of gender representation in Hollywood trailers using computer vision. Detects faces, estimates gender, shot scale, and gaze, computes screen-time and objectification metrics, and generates visualizations and summary reports across genres.

## Overview
This project performs a computational analysis of gender representation in Hollywood trailers. Using computer vision and scene analysis, it quantifies screen-time, shot scale, and gaze direction, generating metrics and visualizations to study gendered patterns across genres.

## Features
- Downloads trailers by genre from YouTube.  
- Samples trailers from the Trailers12k dataset.  
- Scene detection and extraction from trailers.  
- Face detection and gender estimation using InsightFace.  
- Shot scale classification with a ResNet-based CNN.  
- Gaze estimation using MediaPipe Face Mesh.  
- Computes screen-time, close-up rates, and a proxy objectification index.  
- Generates plots and Markdown summary reports.  

## Installation
```bash
git clone https://github.com/<your-username>/gendered-trailers-analysis.git
cd gendered-trailers-analysis
pip install -r requirements.txt
