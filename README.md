
# 🚦 Real-Time Traffic Analysis (Bangalore)

[![GitHub stars](https://img.shields.io/github/stars/Shashwat-19/Real-Time-Traffic-Analysis?style=social)](https://github.com/Shashwat-19/Real-Time-Traffic-Analysis/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org)

---

![BannerBangalore Traffic Visualization Preview](https://github.com/Shashwat-19/Real-Time-Traffic-Analysis/raw/main/Assets/Banner.jpeg)

## 🧭 Overview  
**Real-Time Traffic Analysis** is a machine learning project that leverages the **TomTom Traffic API** to analyze live traffic conditions in Bangalore. It classifies routes as **Congested** or **Normal** using a **Support Vector Machine (SVM)** and visualizes both predictions and real data on an interactive map and performance charts.

---

## 📦 Latest Version: [v1.0](https://github.com/Shashwat-19/Real-Time-Traffic-Analysis/releases/tag/v1.0)  
This version introduces live traffic monitoring, real-time predictions, and a visual map powered by **Folium**.

---

## ✨ Features

- 📡 **Live Traffic Data** — Fetches real-time speed, free-flow speed, and confidence using TomTom API.
- 🤖 **Machine Learning Model** — Uses **SVM** classifier from `scikit-learn` to label traffic as `Normal` or `Congested`.
- 🧮 **Performance Metrics** — View confusion matrix and classification report.
- 📊 **Bar Chart Visualization** — Displays actual vs predicted traffic labels.
- 🗺️ **Interactive Folium Map** — Highlights traffic status with colored markers:
  - 🟢 Green = Normal  
  - 🔴 Red = Congested

---

## 📍 Monitored Locations

```
📌 MG Road  
📌 Whitefield  
📌 Electronic City  
📌 Hebbal  
📌 Yelahanka
```

---

## 🔁 How It Works

The project fetches real-time traffic data from the TomTom API for various Bangalore locations. It extracts features like current speed, free flow speed, and confidence, and labels traffic as either **Normal** or **Congested** based on defined thresholds. An SVM classifier is trained on this data to predict traffic conditions, and results are visualized using charts and an interactive map.

### ✅ Key Steps

- Fetch live traffic data *(speed, free flow speed, confidence)*
- Label data based on speed drop and confidence
- Train and test **SVM classifier**
- Visualize predictions vs actuals in a **bar chart**
- Display traffic on a **folium map** with color-coded markers


---

## 🧪 Try It on Google Colab

✅ No installation hassles  
✅ All dependencies install automatically  
✅ Results update in real-time  
<br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shashwat-19/Real-Time-Traffic-Analysis/blob/main/Traffic_Analysis_Bangalore.ipynb)

---

## 🧰 Tech Stack

- **Language**: Python 🐍  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Folium  
- **API**: TomTom Traffic API  
- **Visualization**: Bar charts & interactive maps

---

## 📁 Project Structure

```
Real-Time-Traffic-Analysis/
├── Traffic_Analysis_Bangalore.ipynb
├── bangalore_traffic_data.csv
├── bangalore_traffic_map.html
├── utils/
│   └── preprocessing.py
├── assets/
│   └── screenshots/
└── README.md
```

---

## 📂 Output Files

| File                          | Description                                  |
|------------------------------|----------------------------------------------|
| `bangalore_traffic_data.csv` | Dataset with traffic features and labels     |
| `bangalore_traffic_map.html` | Interactive traffic map for Bangalore        |
| Inline graphs (in Colab)     | Classification report, confusion matrix etc. |

---

## 🔐 API Key Setup

To use the TomTom API:

1. Sign up at [developer.tomtom.com](https://developer.tomtom.com)
2. Create a project and generate a **free API key**
3. Replace the placeholder in the notebook:
   ```python
   TOMTOM_API_KEY = "YOUR_API_KEY_HERE"
   ```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Shashwat-19/Real-Time-Traffic-Analysis.git
cd Real-Time-Traffic-Analysis

# Open Jupyter Notebook
jupyter notebook Traffic_Analysis_Bangalore.ipynb
```

---

## 📖 Documentation

All code is heavily commented and documented in the notebook itself.  
Additional notes and usage guides will be published on [my blog](https://shashwat-filenest.hashnode.dev/).

---

## 🔒 License
This project is licensed under the **MIT LICENSE**. See the [LICENSE](https://github.com/Shashwat-19/Real-Time-Traffic-Analysis/blob/main/LICENSE) file for details.

---

## 📩 Contact  
### Shashwat  
**Software Developer | Cloud & DevOps Enthusiast**

**🔹 Java Backend Development**<br>
**🔹 Cloud Architecture & Containerization**<br>
**🔹 DevOps & Scalable Systems**

### 🚀 Open Source | Tech Innovation  
Passionate about building scalable applications and contributing to transformative tech solutions.

### 📌 Find me here:  
[<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />](https://github.com/Shashwat-19)  [<img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/shashwatk1956/)  [<img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" />](mailto:shashwat1956@gmail.com)  [<img src="https://img.shields.io/badge/Hashnode-2962FF?style=for-the-badge&logo=hashnode&logoColor=white" />](https://hashnode.com/@Shashwat56)

---

**🔁 Fork it, run it, improve it — let’s make Bangalore traffic predictable together! 🚗**
