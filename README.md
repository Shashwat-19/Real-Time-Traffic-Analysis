
# 🚦 Real-Time Traffic Analysis (Bangalore)

[![GitHub stars](https://img.shields.io/github/stars/Shashwat-19/Real-Time-Traffic-Analysis?style=social)](https://github.com/Shashwat-19/Real-Time-Traffic-Analysis/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org)

---

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

1. Calls **TomTom Traffic API** for each location.
2. Extracts:
   - `Current Speed`
   - `Free Flow Speed`
   - `Confidence`
3. Labels traffic as:
   - `Congested` if speed is significantly lower than free flow speed and confidence is low.
   - `Normal` otherwise.
4. Trains an **SVM Classifier** on preprocessed data.
5. Generates:
   - **Bar chart** of predicted vs actual results
   - **Classification report**
   - **Folium map** with color-coded markers

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

## 📈 Future Enhancements

- 🔄 Automate hourly traffic updates
- 🧠 Try advanced ML models like Random Forest or XGBoost
- 📲 Build a dashboard using Streamlit or Flask

---

## 📝 License

This project is licensed under the **MIT License**.  
See the [LICENSE](https://github.com/Shashwat-19/Real-Time-Traffic-Analysis/blob/main/LICENSE) file for details.

---

## 📩 Contact

### Shashwat  
**Machine Learning Enthusiast | Cloud & DevOps Learner**  
**🔹 Real-Time Data Processing**  
**🔹 Web APIs | Automation | Dashboards**  

---

### 🔗 Connect With Me

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Shashwat-19)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shashwatk1956/)  
[![Gmail](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:shashwat1956@gmail.com)  
[![Hashnode](https://img.shields.io/badge/Hashnode-2962FF?style=for-the-badge&logo=hashnode&logoColor=white)](https://hashnode.com/@Shashwat56)

---

**🔁 Fork it, run it, improve it — let’s make Bangalore traffic predictable together! 🚗**
