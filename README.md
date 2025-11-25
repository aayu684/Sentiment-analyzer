# 🎯 Sentiment Analyzer
This project presents an **Advanced Sentiment Analyzer** designed to accurately interpret sentiment in Hinglish (Hindi–English code-mixed) product reviews and modern Gen Z slang, addressing a major gap in traditional NLP systems. Conventional sentiment analysis tools often fail when confronted with non-standard spellings, hybrid linguistic patterns, and rapidly evolving slang used widely across social media and e-commerce platforms. To overcome these limitations, this project introduces a **hybrid lexicon-based sentiment analysis

---

## 📌 Overview
**Sentiment Analyzer** is a next-generation NLP tool designed to analyze Hinglish (Hindi + English code-mixed text) and Gen Z internet slang, which traditional sentiment analysis systems fail to understand.

### It uses:

- Custom Hinglish lexicons

- Gen Z slang dictionaries

- Negation handling

- Intensity multipliers

- Emotion detection

- Interactive visualizations

**Built with a modern glassmorphism UI, this tool provides detailed polarity scores, confidence metrics, slang intensity, radar charts, emotion bars, and session trend tracking.
(Code reference from app.py used for building this tool.)**

---

## ✨ Key Features
### 🔮 1. Hybrid Sentiment Engine

- Hinglish positive/negative lexicons

- Gen Z slang dictionary (slaps, bussin, W, mid, cringe, etc.)

- Negation-aware scoring (English + Hindi words like not, nahi, mat)

- Intensifier-based weighting: boht, bilkul, hella, crazy, ultra

**Hybrid polarity formula:
80% custom NLP + 20% TextBlob**

### 🎭 2. Emotion Detection

Identifies 4 core emotions:

- 😊 Joy

- 😞 Sadness

- 😡 Anger

- 🤩 Excitement

Provides normalized emotion distribution.

### 🧃 3. Futuristic UI (Glassmorphism + Neon)

- Tailwind CSS injected inside Streamlit

- Liquid animated background blobs

- Gradient headings, smooth shadows

- Clean cards + rounded components

- Custom scrollbar and text fields

### 📊 4. Advanced Data Visualization

- Radar chart for sentiment profile

- Confidence gauge

- Slang intensity meter

- Session polarity trend chart

- CSV export for session analysis

### 🔁 5. Smart History Tracking

- Saves the last few analyses

- Shows mini-preview for each

- Allows CSV export

- One-click clearing

---

## 🛠️ Tech Stack
### ComponenT & Technology
    Frontend	                           Streamlit + TailwindCSS
    NLP Engine	                           TextBlob + custom Hinglish & GenZ lexicons
    Visualization	                       Plotly (Gauge, Radar, Line)
    Programming	                           Python 3.8+
    Data Handling	                       NumPy, Pandas
    Styling	                               Custom CSS + Glassmorphism

---

## 📂 Project Structure
```bash
📁 sentiment-analyzer/
 ├── app.py
 ├── requirements.txt
 ├── README.md
```

---

## 📥 Installation
### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/sentiment-analyzer-pro
cd sentiment-analyzer-pro
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 🧪 Usage Guide
### ▶️ Start the App
Once opened:

1. Enter or paste a product review

2. Choose from quick samples:

- Positive Vibe

- Negative Vibe

- Gen Z Mode

3. Click Analyze Sentiment
### 📊 Output Includes:
1. Final sentiment label (Positive/Negative/Neutral)

2. Emoji + color-coded sentiment card

3. Confidence score gauge

4. Slang intensity score

5. Emotion detection chart

6. Radar chart of:

- Polarity

- Confidence

- Subjectivity

- Slang score

7. Session trend chart

8. CSV download link

---

## 📈 Performance (Benchmark)

Tested on:

- Hinglish reviews

- Slang-heavy Gen Z reviews

- Mixed-code reviews

- Standard English reviews

Achieved:

- Overall accuracy: ~83%

- Far higher accuracy than base TextBlob (48–52%)

---

## 👨‍💻 Author
- **Aayushi soni** – [GitHub](https://github.com/aayu684) | [LinkedIn](https://www.linkedin.com/in/aayushisoni6295/)

---



