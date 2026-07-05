# Agentic Chatbot with Robust Eval Framework

A production-ready AI agent pipeline and comprehensive evaluation framework, built to demonstrate robust conversational routing, guardrails, and LLM-as-a-judge evaluation techniques.

![App Screenshot](https://cdn.corenexis.com/f/UDmDtHwP1Lt.png)



## 🌟 Key Features

### 1. Robust Agent Pipeline (`agents/`)
- **Multi-Intent Router:** Classifies user inputs, assigns confidence scores, and routes to appropriate specialists (Policy, Order, FAQ).
- **Entity Extractor:** Uses fuzzy matching and LLM extraction to pull product names, order IDs, and emails.
- **Safety Guardrails:** Real-time input/output filtering to detect prompt injection, crisis escalation, and data leaks.
- **Synthesizer:** Parallel processing of multi-intent queries, merging responses seamlessly.

### 2. Comprehensive Evaluation Engine (`evals/`)
- **LLM-as-a-Judge:** Uses a strict 6-dimension rubric (Accuracy, Tone, Formatting, Safety, Completeness, Conciseness) to score responses.
- **Semantic Similarity:** Computes cosine similarity between generated responses and reference answers using `sentence-transformers`.
- **Statistical Rigor:** Calculates Spearman correlation, Cohen's d/Kappa, and bootstrap confidence intervals.
- **Bias Detection:** Built-in checks for position bias in judge outputs.

### 3. Streamlit UI (`app/`)
- **Setup & Config:** Dynamically switch between Groq, Gemini, OpenRouter, or local models.
- **Live Chat & Trace:** Chat with the bot and view a real-time debug trace of the entire pipeline (routing, entities, guardrails, latency).
- **Evaluation Runner:** Trigger single or bulk test cases and compare runs (Run A vs Run B).
- **Analytics Dashboard:** Visualize pass rates, score distributions, radar charts of dimensions, and statistical confidence intervals.
- **Data Flywheel:** Promote real chat logs directly into the evaluation dataset.

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- At least one API key (Gemini, Groq, or OpenRouter). Recommended : OpenRouter

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ChatBot+Eval
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` to include your `GEMINI_API_KEY` or `OPENROUTER_API_KEY`.*

4. **Run the Application:**
   ```bash
   streamlit run "app/🛍️_Setup.py"
   ```

5. Navigate to `http://localhost:8502` in your browser.

---

## 🏗️ Project Architecture

```
ChatBot+Eval/
├── app/                  # Streamlit Multi-Page UI
│   ├── 🛍️_Setup.py        # Entry point & Model configuration
│   ├── pages/            # Chat, History, Eval Runner, Dashboard
│   └── assets/           # Midnight Linear custom CSS
├── agents/               # AI Pipeline & Logic
│   ├── router.py         # Intent classification
│   ├── specialists.py    # Domain experts (Order, Policy, FAQ)
│   ├── guardrails.py     # Input/Output safety
│   ├── entity_extractor.py 
│   └── pipeline.py       # Orchestrator & Synthesizer
├── evals/                # Evaluation Framework
│   ├── judge.py          # LLM rubric scoring
│   ├── metrics.py        # Statistics (Spearman, Cohen's d)
│   ├── embeddings.py     # Cosine similarity
│   └── cascade.py        # Master eval orchestrator
├── data/                 # Datasets & Persistence
│   ├── mock_db.json      # Mock store data (users, orders)
│   ├── dataset_*.json    # Eval cases (single, multi, extended)
│   └── eval_results.db   # SQLite storage for logs & metrics
└── Documents/            # Extensive Architectural Documentation
```
*For deep-dive documentation on how data flows through the system, check the `Documents/Architectural Diagram/` folder.*

#### $\textsf{\color{#28a745}{Read the implementation Plan.md (Documents folder)file for more details on running the app and evaluations.}}$

---

## 🧪 Running Evaluations

1. Open the **Setup** page and configure your pipeline model (e.g., Gemini 1.5 Flash).
2. Go to the **Evaluation** page.
3. Select your dataset (e.g., `Single,Multi or Categories`).
4. Click **Run Evaluation**. The eval name follows a syntax as eg( `Run|Model Name|Eval Name|Pass Rate`)
5. Switch to the **Dashboard** page to view the statistical results, radar charts, and confidence intervals.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
