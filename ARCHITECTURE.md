# 🏗️ System Architecture - ZMtech AI Options Calculator Pro

## 📊 High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────┐                                      │
│  │   Streamlit Web App     │                                      │
│  │   (app.py)              │                                      │
│  │                         │                                      │
│  │  • Strategy Builder     │                                      │
│  │  • Payoff Diagrams      │                                      │
│  │  • Real-time Greeks     │                                      │
│  │  • AI Strategy Reports  │                                      │
│  └──────────┬──────────────┘                                      │
│             │                                                     │
└─────────────┼─────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────┐
│           Core Logic & Data Layer            │
├───────────────────────┬──────────────────────┤
│  Data Fetching        │  Analytics Engine    │
│  (yfinance)           │  (scipy/numpy)       │
│                       │                      │
│  • Stock Prices       │  • Black-Scholes     │
│  • Option Chains      │  • Greeks (Δ,Γ,Θ,ν)  │
│  • IV History         │  • Prob. of Profit   │
├───────────────────────┴──────────────────────┤
│             │                                │
└─────────────┼────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────┐
│           AI & Reporting Layer               │
├───────────────────────┬──────────────────────┤
│  Google Gemini AI     │  Report Generator    │
│  (generativeai)       │  (docx/fpdf)         │
│                       │                      │
│  • Strategy Analysis  │  • Word Export       │
│  • Market Sentiment   │  • PDF Export        │
│  • Risk Assessment    │                      │
└───────────────────────┴──────────────────────┘
```

## 🔄 Data Flow
```
1. USER INPUT
   ↓
   Strategy Type (e.g., "Iron Condor") + Ticker ("SPY") + Parameters
   ↓

2. DATA ENRICHMENT LAYER (app.py)
   ↓
   • Fetch Real-time Stock Price (yfinance)
   • Download Option Chain (Calls/Puts)
   • Calculate Implied Volatility (IV) Rank
   ↓

3. ANALYTICS ENGINE (Vectorized Operations)
   ↓
   • Generate Price Range Arrays (numpy)
   • Calculate Payoff for Each Leg
   • Compute Aggregate Greeks (Delta, Gamma, Theta, Vega)
   • Estimate Probability of Profit (Monte Carlo approx)
   ↓

4. VISUALIZATION
   ↓
   Interactive Plotly Payoff Diagram
   ↓

5. AI ANALYSIS INTEGRATION
   ↓
   Structured Prompt Construction:
     [Strategy Metrics + Greeks + Market Context]
   ↓
   Google Gemini AI Processing:
     Generates "Institutional Quality" Investment Memo
   ↓

6. OUTPUT & EXPORT
   ↓
   • Streamlit UI Dashboard
   • Downloadable .docx / .pdf Reports
```

## 🗂️ File Structure & Relationships
```
c:\options\
│
├── 📄 app.py ⭐ MAIN APPLICATION
│   ├── class OptionLeg
│   ├── class OptionStrategy
│   ├── UI Configuration
│   ├── Data Fetching (caching enabled)
│   └── Main Execution Loop
│
├── 📄 report_generator.py → REPORTING ENGINE
│   ├── create_word_report()
│   ├── create_pdf_report()
│   └── sanitize_text()
│
├── 📦 requirements.txt → DEPENDENCIES
│   ├── streamlit (UI)
│   ├── plotly (Charts)
│   ├── yfinance (Data)
│   ├── numpy/scipy (Math)
│   ├── google-generativeai (AI)
│   └── python-docx/fpdf (Exports)
│
└── 📚 Documentation:
    └── ARCHITECTURE.md → System architecture (this file)
```

## 🧩 Component Breakdown

### 1. OptionStrategy Class (app.py)
**Purpose**: Core domain model for options positions.

**Key Methods**:
```python
# Initialize strategy
strategy = OptionStrategy(name="Iron Condor", legs=[...], ...)

# Calculate Profit/Loss across price range
payoff = strategy.calculate_payoff(price_range)
# Returns: numpy array of P&L values

# Get comprehensive metrics
metrics = strategy.get_metrics()
# Returns: {
#   "max_profit": float,
#   "max_loss": float,
#   "breakeven_points": [float],
#   "risk_reward_ratio": float
# }
```

### 2. Streamlit Web App (app.py)
**Layout**:
```
┌─────────────────────────────────────────────────────────┐
│  Header: "ZMtech AI Options Calculator Pro"            │
├───────────┬─────────────────────────────────────────────┤
│           │  Strategy Configurator:                     │
│ Sidebar:  │  ┌─────────────────────────────────────┐   │
│           │  │ Category: [Spreads]                 │   │
│ • Config  │  │ Strategy: [Iron Condor]             │   │
│ • Live    │  └─────────────────────────────────────┘   │
│   Data    │  Leg 1: Sell Put @ $95                     │
│   Toggle  │  Leg 2: Buy Put @ $90                      │
│           │  ...                                       │
│           │                                             │
│           │  Payoff Diagram (Interactive):              │
│           │  ┌─────────────────────────────────────┐   │
│           │  │           /---\                     │   │
│           │  │          /     \                    │   │
│           │  │_________/_______\_________          │   │
│           │  └─────────────────────────────────────┘   │
│           │                                             │
│           │  Key Metrics:                               │
│           │  ┌──────────┬──────────┬──────────┐        │
│           │  │Max Profit│ Max Loss │ Risk/Rew │        │
│           │  └──────────┴──────────┴──────────┘        │
│           │                                             │
│           │  AI Analysis:                               │
│           │  ┌─────────────────────────────────────┐   │
│           │  │ 🤖 Gemini Analysis                  │   │
│           │  │ "Bullish sentiment detected..."     │   │
│           │  └─────────────────────────────────────┘   │
│           │  [Download Text | Word | PDF]               │
└───────────┴─────────────────────────────────────────────┘
```

### 3. Analytics & AI Modules
**Purpose**: Advanced math and intelligence layer.

**Key Components**:
*   **Black-Scholes**: `black_scholes_price()` calculates theoretical option prices.
*   **Greeks Engine**: `calculate_strategy_greeks()` aggregates Delta, Gamma, Theta, etc., for the entire multi-leg position.
*   **AI Generator**: `generate_ai_analysis()` bridges the quantitative data with qualitative insights from Google Gemini.

## 🔐 Security & Configuration
*   **API Keys**: Google Gemini API key is managed via `st.secrets` or environment variables for security.
*   **Input Validation**: robust error handling for ticker symbols and numerical inputs.

## 📊 Data Schema
**Option Leg Structure**:
```json
{
  "type": "call" | "put",
  "action": "buy" | "sell",
  "strike": float,
  "premium": float,
  "quantity": int,
  "expiration_days": int
}
```

## ⚡ Performance Optimization
*   **Caching**: `@st.cache_data` is used for:
    *   Stock Data (30 min TTL)
    *   Option Chains (30 min TTL)
    *   Expiration Dates (1 hour TTL)
*   **Vectorization**: `numpy` is used for all heavy P&L calculations to ensure instant payoff diagram rendering.
