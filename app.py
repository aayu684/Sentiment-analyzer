import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from datetime import datetime
import numpy as np
import base64

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Tailwind CSS
st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        /* Custom Font */
        * {
            font-family: 'Outfit', sans-serif !important;
        }

        /* Liquid Background Animation */
        .liquid-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #0f172a;
            z-index: -1;
            overflow: hidden;
        }

        .blob {
            position: absolute;
            filter: blur(80px);
            opacity: 0.6;
            animation: float 10s infinite ease-in-out;
        }

        .blob-1 {
            top: -10%;
            left: -10%;
            width: 500px;
            height: 500px;
            background: #4f46e5;
            animation-delay: 0s;
        }

        .blob-2 {
            bottom: -10%;
            right: -10%;
            width: 600px;
            height: 600px;
            background: #7c3aed;
            animation-delay: 2s;
        }

        .blob-3 {
            top: 40%;
            left: 40%;
            width: 400px;
            height: 400px;
            background: #db2777;
            animation-delay: 4s;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(30px, -50px) scale(1.1); }
            66% { transform: translate(-20px, 20px) scale(0.9); }
            100% { transform: translate(0, 0) scale(1); }
        }

        /* Glassmorphism Utilities */
        .glass {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }

        .glass-hover:hover {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        }

        /* Streamlit Overrides */
        .stApp {
            background: transparent;
        }
        
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 12px;
        }
        
        .stTextArea textarea:focus {
            border-color: #8b5cf6 !important;
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3) !important;
        }

        /* Hide Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02); 
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1); 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2); 
        }
    </style>
    
    <!-- Liquid Background Elements -->
    <div class="liquid-bg">
        <div class="blob blob-1"></div>
        <div class="blob blob-2"></div>
        <div class="blob blob-3"></div>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'sample_review' not in st.session_state:
    st.session_state.sample_review = ""

# --- Dictionaries ---
HINGLISH_POSITIVE = {
    'mast': 1.0, 'badhiya': 1.0, 'zabardast': 1.0, 'kamaal': 1.0, 'shandar': 1.0,
    'badiya': 1.0, 'accha': 0.8, 'acha': 0.8, 'achha': 0.8, 'achchha': 0.8,
    'sahi': 0.7, 'ekdum': 0.8, 'dhansu': 1.0, 'jhakaas': 1.0, 'lajawaab': 1.0, 
    'top': 0.8, 'best': 1.0, 'maja': 0.8, 'mazaa': 0.8, 'maza': 0.8, 
    'superb': 1.0, 'badass': 0.9, 'dope': 0.9, 'lit': 0.9, 'fire': 0.9, 
    'op': 0.8, 'behtreen': 1.0, 'gazab': 0.9, 'ghazab': 0.9, 'tagda': 0.8, 
    'solid': 0.8, 'perfect': 1.0, 'bindas': 0.8, 'kadak': 0.8, 'fadu': 0.9, 
    'faadu': 0.9, 'jhakkas': 1.0, 'sunder': 0.8, 'sundar': 0.8, 'pyara': 0.7, 
    'pyaara': 0.7, 'pasand': 0.7, 'like': 0.7, 'laga': 0.6, 'lagaa': 0.6,
    'good': 0.8, 'nice': 0.7, 'great': 0.9, 'awesome': 0.9, 'excellent': 1.0
}

HINGLISH_NEGATIVE = {
    'bekaar': -0.9, 'bekar': -0.9, 'ganda': -0.8, 'kharab': -0.9, 'bakwas': -1.0,
    'faltu': -0.8, 'bakwaas': -1.0, 'ghatiya': -1.0, 'wahiyat': -1.0, 'bura': -0.8,
    'kachra': -1.0, 'waste': -0.9, 'bakkar': -0.9, 'thanda': -0.6, 'boring': -0.7,
    'chutiya': -1.0, 'chutiyapa': -1.0, 'bekuf': -0.8, 'bevkuf': -0.8,
    'pagal': -0.5, 'pagalpan': -0.6, 'nautanki': -0.6, 'dramabaazi': -0.6,
    'dhokha': -0.9, 'fraud': -1.0, 'scam': -1.0, 'locha': -0.7, 'problem': -0.6,
    'bad': -0.8, 'worst': -1.0, 'horrible': -1.0, 'terrible': -1.0, 'poor': -0.7
}

GENZ_SLANG_POSITIVE = {
    'slaps': 1.0, 'slap': 1.0, 'bussin': 1.0, 'fire': 0.9, 'lit': 0.9, 'dope': 0.9,
    'goat': 1.0, 'slay': 0.9, 'slaying': 0.9, 'iconic': 0.9, 'vibe': 0.7, 'vibes': 0.7,
    'chef': 0.8, 'chefs': 0.8, 'kiss': 0.8, 'snack': 0.7, 'queen': 0.8, 'king': 0.8,
    'ate': 0.9, 'no': 0.0, 'cap': 0.8, 'based': 0.8, 'w': 0.9, 'bet': 0.7,
    'facts': 0.7, 'fr': 0.7, 'frfr': 0.8, 'goated': 1.0, 'hits': 0.8, 'different': 0.8,
    'built': 0.7, 'valid': 0.8, 'understood': 0.8, 'assignment': 0.9, 'serving': 0.8,
    'stan': 0.7, 'bop': 0.8, 'banger': 0.9, 'smash': 0.8, 'bussing': 1.0
}

GENZ_SLANG_NEGATIVE = {
    'mid': -0.6, 'cringe': -0.8, 'cringy': -0.8, 'cringing': -0.8, 'ick': -0.7,
    'sus': -0.6, 'sussy': -0.7, 'cap': -0.8, 'capping': -0.8, 'yikes': -0.7,
    'bruh': -0.4, 'oof': -0.5, 'l': -0.8, 'ratio': -0.7, 'ratiod': -0.8,
    'trash': -0.9, 'flop': -0.8, 'flopped': -0.8, 'dead': -0.7, 'awkward': -0.6,
    'weird': -0.5, 'pressed': -0.6, 'salty': -0.6, 'toxic': -0.9, 'cancelled': -0.9,
    'cringe': -0.8, 'basic': -0.5, 'tryhard': -0.6, 'dry': -0.6, 'boring': -0.7
}

HINDI_INTENSIFIERS = {
    'boht': 1.5, 'bohot': 1.5, 'bahut': 1.5, 'bahot': 1.5, 'bht': 1.5,
    'ekdum': 1.4, 'bilkul': 1.3, 'poora': 1.2, 'pura': 1.2, 'kaafi': 1.3,
    'full': 1.3, 'very': 1.3, 'so': 1.2, 'too': 1.2, 'super': 1.4,
    'ultra': 1.5, 'mega': 1.4, 'hella': 1.4, 'mad': 1.3, 'crazy': 1.3
}

# Emotion Keywords (Heuristics)
EMOTION_KEYWORDS = {
    'joy': ['happy', 'love', 'great', 'awesome', 'mast', 'best', 'fun', 'enjoy', 'smile', 'laugh', 'lit', 'fire', 'slay'],
    'sadness': ['sad', 'bad', 'cry', 'depressed', 'unhappy', 'worst', 'poor', 'bekaar', 'trash', 'dead', 'flop'],
    'anger': ['angry', 'hate', 'mad', 'furious', 'annoyed', 'irritated', 'bakwas', 'ghatiya', 'cringe', 'toxic'],
    'excitement': ['excited', 'wow', 'amazing', 'omg', 'hype', 'crazy', 'dhansu', 'jhakaas', 'goat', 'bussin']
}

# --- Functions ---

def clean_text(text):
    """Clean and preprocess text while preserving Hinglish"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s!?.,\-]', '', text, flags=re.UNICODE)
    return text.strip()

def analyze_emotions(text):
    """Simple heuristic-based emotion detection"""
    words = text.lower().split()
    emotions = {'joy': 0, 'sadness': 0, 'anger': 0, 'excitement': 0}
    
    for word in words:
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if word in keywords:
                emotions[emotion] += 1
                
    # Normalize
    total = sum(emotions.values())
    if total > 0:
        for k in emotions:
            emotions[k] /= total
    return emotions

def analyze_hinglish_genz_sentiment(text):
    """Enhanced sentiment analysis for Hinglish and Gen Z slang"""
    cleaned_text = clean_text(text)
    words = cleaned_text.lower().split()
    
    blob = TextBlob(cleaned_text)
    base_polarity = blob.sentiment.polarity
    base_subjectivity = blob.sentiment.subjectivity
    
    negation_words = ['nahi', 'nai', 'nhi', 'na', 'mat', 'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'none', 'nahin']
    
    custom_score = 0
    word_count = 0
    slang_count = 0
    intensifier_multiplier = 1.0
    skip_indices = set()
    
    for i, word in enumerate(words):
        if i in skip_indices:
            continue
            
        sentiment_value = None
        word_type = None
        
        if word in HINGLISH_POSITIVE:
            sentiment_value = HINGLISH_POSITIVE[word]
            word_type = 'positive'
            slang_count += 1
        elif word in HINGLISH_NEGATIVE:
            sentiment_value = HINGLISH_NEGATIVE[word]
            word_type = 'negative'
            slang_count += 1
        elif word in GENZ_SLANG_POSITIVE:
            sentiment_value = GENZ_SLANG_POSITIVE[word]
            word_type = 'positive'
            slang_count += 1
        elif word in GENZ_SLANG_NEGATIVE:
            sentiment_value = GENZ_SLANG_NEGATIVE[word]
            word_type = 'negative'
            slang_count += 1
        
        if sentiment_value is not None:
            is_negated = False
            for j in range(max(0, i-3), i):
                if words[j] in negation_words:
                    is_negated = True
                    break
            for j in range(i+1, min(len(words), i+4)):
                if words[j] in negation_words:
                    is_negated = True
                    skip_indices.add(j)
                    break
            
            if i > 0 and words[i-1] in HINDI_INTENSIFIERS:
                intensifier_multiplier = HINDI_INTENSIFIERS[words[i-1]]
            
            score = sentiment_value * intensifier_multiplier
            if is_negated:
                score = -abs(score) if word_type == 'positive' else abs(score) * 0.6
            
            custom_score += score
            word_count += 1
            intensifier_multiplier = 1.0
        
        elif word == 'no' and i + 1 < len(words) and words[i + 1] == 'cap':
            custom_score += 0.8
            word_count += 1
            slang_count += 1
            skip_indices.add(i + 1)
    
    if word_count > 0:
        custom_polarity = custom_score / max(word_count, 1)
        final_polarity = (custom_polarity * 0.8 + base_polarity * 0.2)
    else:
        final_polarity = base_polarity
    
    if final_polarity > 0.15:
        sentiment = "Positive"
        emoji = "😊"
        color = "#10b981"
    elif final_polarity < -0.15:
        sentiment = "Negative"
        emoji = "😞"
        color = "#ef4444"
    else:
        sentiment = "Neutral"
        emoji = "😐"
        color = "#f59e0b"
    
    confidence = min(abs(final_polarity) * 100, 95)
    if word_count > 0:
        confidence = min(confidence + 10, 98)
    
    emotions = analyze_emotions(cleaned_text)
    
    return {
        'sentiment': sentiment,
        'emoji': emoji,
        'color': color,
        'polarity': final_polarity,
        'subjectivity': base_subjectivity,
        'confidence': confidence,
        'slang_detected': slang_count > 0,
        'slang_score': min(slang_count * 20, 100),
        'emotions': emotions
    }

def create_radar_chart(data):
    """Create a radar chart for sentiment profile"""
    categories = ['Polarity', 'Subjectivity', 'Confidence', 'Slang Score']
    
    # Normalize values to 0-100 scale for chart
    values = [
        (data['polarity'] + 1) * 50,  # Convert -1..1 to 0..100
        data['subjectivity'] * 100,
        data['confidence'],
        data['slang_score']
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(139, 92, 246, 0.3)',
        line=dict(color='#8b5cf6', width=2),
        name='Current Review'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='rgba(255,255,255,0.5)'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(color='#e2e8f0', size=12),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=False
    )
    return fig

def create_trend_chart(history):
    """Create a trend chart from history"""
    if not history:
        return None
        
    df = pd.DataFrame(history)
    df['index'] = range(len(df))
    
    fig = px.line(df, x='index', y='polarity', markers=True)
    fig.update_traces(line_color='#6366f1', line_width=3, marker=dict(size=8, color='#c084fc'))
    
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, title=None),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.2)', title='Polarity'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=0, r=0, t=10, b=10),
        height=200
    )
    return fig

def get_csv_download_link(history):
    """Generate a link to download history as CSV"""
    if not history:
        return ""
    df = pd.DataFrame(history)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_history.csv" class="text-sm text-indigo-400 hover:text-indigo-300 underline decoration-indigo-500/30">Download CSV</a>'
    return href

# --- Layout ---

# Sidebar
with st.sidebar:
    st.markdown("""
        <div class="glass p-6 rounded-2xl mb-6">
            <h2 class="text-xl font-bold text-white mb-2">📜 History</h2>
            <div class="h-1 w-12 bg-indigo-500 rounded-full mb-4"></div>
    """, unsafe_allow_html=True)
    
    if st.session_state.history:
        # Actions
        col_d, col_c = st.columns([1, 1])
        with col_d:
            st.markdown(get_csv_download_link(st.session_state.history), unsafe_allow_html=True)
        with col_c:
            if st.button("Clear", key="clear_hist", type="secondary"):
                st.session_state.history = []
                st.rerun()
        
        st.markdown('<div class="space-y-3 mt-4">', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            color_class = "border-green-500" if item['sentiment'] == 'Positive' else "border-red-500" if item['sentiment'] == 'Negative' else "border-yellow-500"
            st.markdown(f"""
                <div class="glass p-3 rounded-xl border-l-4 {color_class} transition hover:bg-white/5">
                    <div class="flex justify-between items-center mb-1">
                        <span class="text-xs text-slate-400">{item['timestamp']}</span>
                        <span class="text-xs font-bold text-white">{item['sentiment']}</span>
                    </div>
                    <p class="text-sm text-slate-300 line-clamp-2">{item['text']}</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="text-slate-500 text-sm italic">No analysis yet.</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main Content
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    # Hero
    st.markdown("""
        <div class="text-center py-12 animate-fade-in-up">
            <h1 class="text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 mb-4 drop-shadow-lg">
                Product Review Sentiment Analyzer
            </h1>
            <p class="text-xl text-slate-300 font-light tracking-wide">
                Next-Gen NLP for <span class="text-indigo-400 font-medium">Hinglish</span> & <span class="text-pink-400 font-medium">Gen Z Slang</span>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    st.markdown('<div class="glass rounded-3xl p-8 mb-8 glass-hover transition-all duration-300">', unsafe_allow_html=True)
    
    # Quick Actions
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✨ Positive Vibe", key="btn_pos", use_container_width=True):
            st.session_state.sample_review = "Boht mast product hai! Quality ekdum top notch. This slaps fr! 🔥"
            st.rerun()
    with c2:
        if st.button("💀 Negative Vibe", key="btn_neg", use_container_width=True):
            st.session_state.sample_review = "Total waste of money. Bekaar quality, huge L. Cringe experience."
            st.rerun()
    with c3:
        if st.button("🤪 Gen Z Mode", key="btn_genz", use_container_width=True):
            st.session_state.sample_review = "No cap this is goated! W purchase, hits different. Bussin fr fr!"
            st.rerun()
            
    st.markdown('<div class="mt-6">', unsafe_allow_html=True)
    review_text = st.text_area(
        "Input",
        value=st.session_state.sample_review,
        height=120,
        placeholder="Type something... (e.g. 'Kya mast cheez hai!')",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        if review_text:
            with st.spinner("Crunching numbers..."):
                result = analyze_hinglish_genz_sentiment(review_text)
                
                # Update History
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime("%H:%M"),
                    'text': review_text,
                    'sentiment': result['sentiment'],
                    'polarity': result['polarity'],
                    'confidence': result['confidence'],
                    'slang_score': result['slang_score'],
                    'subjectivity': result['subjectivity']
                })
                
                st.markdown('</div>', unsafe_allow_html=True) # Close input card
                
                # Results Dashboard
                st.markdown('<div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">', unsafe_allow_html=True)
                
                # Card 1: Main Sentiment
                st.markdown(f"""
                    <div class="glass rounded-3xl p-6 text-center flex flex-col justify-center items-center glass-hover">
                        <div class="text-6xl mb-2 animate-bounce">{result['emoji']}</div>
                        <h2 class="text-3xl font-bold text-white mb-1">{result['sentiment']}</h2>
                        <div class="text-sm text-slate-400 uppercase tracking-wider font-semibold">Verdict</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Card 2: Confidence Gauge (Plotly)
                with st.container():
                    st.markdown('<div class="glass rounded-3xl p-4 h-full glass-hover">', unsafe_allow_html=True)
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = result['confidence'],
                        title = {'text': "Confidence", 'font': {'size': 14, 'color': '#94a3b8'}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 0, 'tickcolor': "darkblue"},
                            'bar': {'color': result['color']},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 0,
                            'steps': [{'range': [0, 100], 'color': 'rgba(255,255,255,0.05)'}],
                        },
                        number = {'font': {'color': 'white'}}
                    ))
                    fig_gauge.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Card 3: Slang Score
                st.markdown(f"""
                    <div class="glass rounded-3xl p-6 text-center flex flex-col justify-center items-center glass-hover">
                        <div class="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-br from-pink-500 to-orange-400 mb-2">
                            {result['slang_score']}%
                        </div>
                        <div class="text-white font-bold text-lg">Slang Score</div>
                        <div class="text-xs text-slate-400 mt-2">
                            {'🔥 High Slang' if result['slang_score'] > 50 else '🧊 Low Slang'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True) # End Grid
                
                # Advanced Analytics Row
                c_left, c_right = st.columns([1, 1])
                
                with c_left:
                    st.markdown('<div class="glass rounded-3xl p-6 glass-hover h-full">', unsafe_allow_html=True)
                    st.markdown('<h3 class="text-lg font-bold text-white mb-4">📊 Sentiment Profile</h3>', unsafe_allow_html=True)
                    radar_fig = create_radar_chart(result)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with c_right:
                    st.markdown('<div class="glass rounded-3xl p-6 glass-hover h-full">', unsafe_allow_html=True)
                    st.markdown('<h3 class="text-lg font-bold text-white mb-4">🧠 Emotion Detect</h3>', unsafe_allow_html=True)
                    
                    emotions = result['emotions']
                    if sum(emotions.values()) == 0:
                        st.markdown('<div class="text-center text-slate-500 py-10">No specific emotions detected.</div>', unsafe_allow_html=True)
                    else:
                        for emotion, score in emotions.items():
                            if score > 0:
                                width = int(score * 100)
                                color = {
                                    'joy': 'bg-yellow-400',
                                    'sadness': 'bg-blue-400',
                                    'anger': 'bg-red-400',
                                    'excitement': 'bg-purple-400'
                                }.get(emotion, 'bg-gray-400')
                                
                                st.markdown(f"""
                                    <div class="mb-4">
                                        <div class="flex justify-between text-sm text-slate-300 mb-1">
                                            <span class="capitalize">{emotion}</span>
                                            <span>{width}%</span>
                                        </div>
                                        <div class="w-full bg-slate-700 rounded-full h-2.5">
                                            <div class="{color} h-2.5 rounded-full" style="width: {width}%"></div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Trend Chart (if history > 1)
                if len(st.session_state.history) > 1:
                    st.markdown('<div class="glass rounded-3xl p-6 mt-6 glass-hover">', unsafe_allow_html=True)
                    st.markdown('<h3 class="text-lg font-bold text-white mb-4">📈 Session Trend</h3>', unsafe_allow_html=True)
                    trend_fig = create_trend_chart(st.session_state.history)
                    st.plotly_chart(trend_fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.warning("⚠️ Please enter some text first!")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('</div>', unsafe_allow_html=True) # Close input card if not clicked

# Footer
st.markdown("""
    <div class="text-center py-8 text-slate-500 text-sm">
        <p>Build by Aayushi soni and Ishitaba umat</p>
    </div>
""", unsafe_allow_html=True)