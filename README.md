# 🎬 Analyzing Gender Representation in Film Through Computational

Computational analysis of gender representation in Hollywood trailers using computer vision.  
The pipeline detects faces, estimates gender, classifies shot scale, estimates gaze, and computes screen-time and objectification metrics across genres.  

---

## 📖 Overview  
This project performs a computational study of **gender representation in Hollywood movie trailers**.  
Using computer vision and scene analysis, it quantifies:  

- Screen-time distribution by gender  
- Shot scale (close-up, medium, wide)  
- Gaze direction (up, down, left, right, center)  
- A proxy **objectification index** based on shot scale & gaze  

The workflow downloads trailers, processes them by genre, and outputs JSON results and summary statistics.  

---

## ✨ Features  
- ✅ Download trailers by genre from YouTube  
- ✅ Sample trailers from the **Trailers12k** dataset  
- ✅ Automatic scene detection and segmentation  
- ✅ Face detection & gender classification (InsightFace)  
- ✅ Shot scale classification (ResNet18 CNN)  
- ✅ Gaze estimation (MediaPipe Face Mesh)  
- ✅ JSON results + aggregated statistics  
- ✅ Ready for visualization & reporting  

---

## 📂 Data Availability  
This project uses the **Trailers12k dataset** (metadata and YouTube trailer IDs).  
The dataset is **not included in this repository** due to size constraints.  

To reproduce the analysis:  
1. Download `metadata.tar.gz` from the official Trailers12k dataset website:  
   👉 https://zenodo.org/records/5716410
2. Extract it to obtain `metadata.json`.  
3. Place `metadata.json` in your working directory.  
4. Run extract_genre_trailers.py to generate per-genre trailer IDs and sampled CSVs.  

---

## ⚙️ Installation 

```bash
git clone https://github.com/Medahmedou99/gendered-trailers-analysis.git
cd hollywood-trailers-gender-analysis
pip install -r requirements.txt
