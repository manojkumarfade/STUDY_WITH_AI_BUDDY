# 📚 AI Study Buddy – Enhanced

> An intelligent, gamified, and emotionally aware **AI-powered learning assistant** built with Streamlit.  
> Designed to make **self-study engaging, structured, and fun** — powered by **Gemini + A4F AI**, and enhanced with **gamification, web search, and adaptive learning**.

---

## 🌟 Overview

**AI Study Buddy – Enhanced** isn’t just another chatbot.  
It’s a full-fledged **AI learning companion** built to help students who struggle with self-discipline, motivation, or focus while studying alone.

This app understands *how* you study, adapts to *your style*, and keeps you *motivated* using XP, streaks, achievements, and even a **virtual study pet** that evolves with your progress.  

Whether you’re a student, lifelong learner, or someone trying to master a new topic — AI Study Buddy gives you structure, encouragement, and accountability, all powered by modern AI.

---

## 🎯 Why This Project Matters

Many students struggle with:
- Lack of motivation or consistency in self-study
- Feeling isolated without feedback or guidance
- Not knowing *what* to focus on or *how* to study effectively
- Procrastination or burnout

**AI Study Buddy** solves these by combining:
✅ **AI coaching** to explain concepts in multiple styles  
✅ **Gamification** to reward consistency and progress  
✅ **Emotional intelligence** to encourage during tough times  
✅ **Adaptive tools** like flashcards, quizzes, and personalized learning plans  

It’s not just about studying harder — it’s about studying *smarter*.

---

## 🧩 Key Features

### 🔐 Authentication & User System
- Secure **email/password registration**
- **Email verification** via Gmail SMTP (OTP code)
- Persistent login with **15-minute auto-expiry session**
- **Account deletion** and data cleanup
- Full **SQLite-based database** schema with migration support

---

### 💬 AI Chat Assistant
- Dual AI Integration:
  - **Gemini API** (Google)
  - **A4F API** (Llama 3-based, OpenAI-compatible)
- **Automatic fallback** if one API fails  
- Supports **multiple personas**:
  - 👨‍🏫 Professor — detailed, structured responses  
  - 💡 Socratic — teaches by questioning  
  - 👶 ELI5 — simplifies for beginners  
  - ⚡ Speed — concise and to the point  
  - 🔥 Motivator — boosts morale and focus  
- Web search using **Serper API** for real-time context  
- Memory-aware responses that recall past conversations  
- Emotion detection with **supportive feedback**  

---

### 🧠 Learning & Productivity Tools
- **Flashcards** with spaced repetition  
- **Quizzes** with XP rewards  
- **Study sessions** and streak tracking  
- **Pomodoro timer** for focused sessions  
- **Daily quests** and learning goals  
- **Learning style detection** (Visual, Auditory, Kinesthetic)

---

### 🕹️ Gamification System
- **XP and Level system** for measurable progress  
- **Achievements** like:
  - 🎯 First Study Session  
  - 🔥 7-Day Streak  
  - 💯 Perfect Quiz Score  
- **Study Pet** that evolves with learning:
  - 🥚 Egg → 🐲 Baby Dragon → 🐉 Dragon  
  - Happiness meter based on study activity  
- **Daily rewards** for consistency  

---

### 🧩 Advanced AI Features
- OCR text extraction via **Tesseract + OpenCV**
- **Speech-to-text** input (microphone-based)
- **YouTube Transcript Summarizer**
- **AI Image Generation** (Diffusers / Transformers)
- **YOLO-based Object Detection**
- **3D Visualization** using PyVista
- **Webcam integration** with Streamlit WebRTC

---

### 🧰 Developer Utilities
- Auto schema creation and migration  
- Debug tools: view user data and schema structure  
- **Reset / Force Schema Update** buttons  
- Built-in logging and session cleanup  

---

## 🧱 Tech Stack

| Category | Technologies Used |
|-----------|-------------------|
| **Frontend** | Streamlit |
| **Backend / Logic** | Python |
| **Database** | SQLite |
| **AI APIs** | Google Gemini API, A4F API |
| **Web Search** | Serper API |
| **AI / ML Tools** | Transformers, Diffusers, YOLO, PyTorch |
| **Email System** | Gmail SMTP (SSL) |
| **Computer Vision** | OpenCV, Tesseract OCR |
| **Speech & Media** | SpeechRecognition, PyVista, Streamlit WebRTC |

---

## 🧩 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/AI-Study-Buddy-Enhanced.git
cd AI-Study-Buddy-Enhanced
