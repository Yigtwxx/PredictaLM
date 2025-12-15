# PredictaLM 🧠

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**A lightweight, modern, and interactive interface for experimenting with MiniGPT-based Language Models.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Contact](#-contact)

</div>


---

## 🚀 Overview

**PredictaLM** is a sophisticated web application designed to demonstrate the capabilities of small-scale Generative Pre-trained Transformers (GPT). It provides a sleek, glassmorphism-inspired user interface for users to interact with a custom-trained MiniGPT model, visualize token generation in real-time, and manage their favorite prompts.

Built with **FastAPI** for high-performance backend processing and vanilla **HTML/CSS/JS** for a lightweight, responsive frontend, PredictaLM bridges the gap between complex neural network operations and user-friendly design.

## 🖼️ PredictaLM UI
 <img width="1600" height="900" alt="Ekran görüntüsü 2025-11-23 012042" src="https://github.com/user-attachments/assets/3c643000-b23f-46a8-9ec7-7dae56cc16a7" />



</div>

## ✨ Features

- **🧠 Neural Network Integration:** Powered by a custom MiniGPT implementation based on the Transformer architecture.
- **⚡ Real-Time Inference:** Experience low-latency text generation with immediate feedback.
- **🎨 Modern UI/UX:** A premium, dark-themed interface featuring glassmorphism effects, smooth animations, and responsive design.
- **💾 Persistent Storage:** Save your favorite generations, view history, and manage saved items using a built-in SQLite database.
- **👻 Ghost Text Prediction:** Innovative "ghost text" feature that visualizes the model's thought process as you type.
- **🛠️ Full-Stack Architecture:** A robust separation of concerns with a Python/FastAPI backend and a modular frontend.

## 🏗️ Architecture

PredictaLM follows a modular 3-tier architecture:

1.  **Frontend (UI Layer):**
    *   Vanilla JavaScript for asynchronous API communication.
    *   Custom CSS for a high-fidelity visual experience.
    *   Dynamic DOM manipulation for real-time updates.

2.  **Backend (Application Layer):**
    *   **FastAPI:** Handles HTTP requests, routing, and API documentation.
    *   **Model Engine:** Manages the lifecycle of the MiniGPT model (loading, inference, tokenization).

3.  **Data Layer (Persistence):**
    *   **SQLite & SQLAlchemy:** Manages structured data storage for logs and saved user items.
    *   **Neural Weights:** Stores the trained parameters of the Transformer model.

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Yigtwxx/PredictaLM.git
    cd PredictaLM
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Model Files**
    *   Ensure your trained model checkpoint (`model_best.pt`) is placed in `outputs/checkpoints/`.
    *   Ensure your tokenizer configuration (`tokenizer.json`) is placed in `outputs/tokenizer/`.

## ▶️ Usage

1.  **Start the Application**
    ```bash
    python src/app.py
    ```

2.  **Access the Interface**
    *   The application will automatically open in your default web browser.
    *   If not, navigate to: `http://localhost:7860/ui`

3.  **Interact**
    *   Type a prompt in the input box.
    *   Click Run to generate text.
    *   Click Save to store interesting results.
    *   Use the sidebar to view and manage your saved history.

## 📂 Project Structure

A detailed look at the codebase organization:

```
PredictaLM/
├── db/                         # 🗄️ Database Layer
│   ├── models.py               # SQLAlchemy models (SavedItem, GenerationLog)
│   └── session.py              # Database connection and session management
│
├── outputs/                    # 💾 Model Artifacts
│   ├── checkpoints/            # Trained model weights (model_best.pt)
│   └── tokenizer/              # Tokenizer configuration (tokenizer.json)
│
├── src/                        # 🧠 Source Code (Backend & AI)
│   ├── app.py                  # 🚀 Main FastAPI application & API endpoints
│   ├── model.py                # 🤖 MiniGPT Transformer Neural Network 
│   ├── tokenizer.py            # 🔡 Custom Tokenizer logic
│   ├── train.py                # 🏋️‍♂️ Training script for the model
│   ├── dataset.py              # 📊 Data loading and processing utilities
│   ├── generate.py             # ⚡ CLI script for text generation
│   └── plot_metrics.py         # 📈 Visualization tools for training metrics
│
├── ui/                         # 🎨 Frontend (User Interface)
│   ├── index.html              # 📄 Main HTML structure
│   ├── style.css               # 💅 Custom CSS (Glassmorphism, Dark Mode)
│   └── app.js                  # ⚡ JavaScript logic & API integration
│
├── requirements.txt            # 📦 Python dependencies
└── README.md                   # 📖 Project documentation
```
---

## 📜 License

Released under the **MIT License**. See `LICENSE` for details.

## 🤝 Contact & Connect

---

## 💬 Author

**Yiğit Erdoğan (Yigtwxx)**
📧 [yigiterdogan6@icloud.com](mailto:yigiterdogan6@icloud.com)


🧠 Focus Areas: Deep Learning • Computer Vision • Data Science

---
LinkedIn: [Yiğit ERDOĞAN](www.linkedin.com/in/yiğit-erdoğan-ba7a64294)

--- 

<div align="center">
  <sub>Built with ❤️ using Python and Deep Learning</sub>
</div>
