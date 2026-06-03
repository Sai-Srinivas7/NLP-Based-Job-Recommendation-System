# NLP-Based Job Recommendation System

A machine learning system that analyzes resumes, predicts suitable job roles, and identifies skill gaps. Built as a final project for CSCI 6443 Data Mining (Spring 2025) at George Washington University.


## Overview

The system is trained on 32,000+ job postings from Naukri.com. Job titles are normalized and grouped into canonical categories (e.g. software_engineer, data_scientist, ml_engineer). Skills from each posting are vectorized using TF-IDF, and an SGD-based logistic regression classifier learns the mapping from skill sets to job roles.

At inference time, a user submits a resume (paste or PDF upload). GPT-4o-mini extracts skills from the resume text, the trained classifier ranks job roles by probability, and a skill gap analysis shows which skills the user has vs. what each role expects.

Key metrics on the test set: 60.3% top-1 accuracy, 97.1% top-5 accuracy, 56.4% weighted F1.

## Project Structure

```
.
├── data/
│   └── Naukri Jobs Data.csv      # Training dataset
├── model/
│   ├── chain.py                  # Streamlit app (frontend + inference)
│   ├── recommender.py            # Model training script
│   ├── python.py                 # API key loader (reads from .env)
│   ├── .env.example              # Template for environment variables
│   └── requirements.txt          # Python dependencies
├── project.ipynb                 # EDA and model development notebook
├── datamining_final paper.pdf    # Final project paper
└── project Colab.pdf             # Notebook export
```

## Setup

Requires Python 3.8+.

**1. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**2. Install dependencies**

```bash
cd model
pip install -r requirements.txt
```

**3. Configure your OpenAI API key**

```bash
cp .env.example .env
```

Open `.env` and set your key:

```
OPENAI_API_KEY=your_key_here
```

Do not commit the `.env` file — it is already in `.gitignore`.

**4. Train the model**

```bash
python recommender.py
```

This produces `recommender.pkl` (the trained classifier and TF-IDF vectorizer).

**5. Run the app**

```bash
streamlit run chain.py
```

The app opens in your browser. Paste resume text or upload a PDF to get job recommendations with skill gap analysis.
