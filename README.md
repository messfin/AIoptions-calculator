# 📈 Options Profit Calculator Pro

A comprehensive options trading calculator built with Streamlit and powered by Google Generative AI. This application helps traders visualize and analyze various options strategies with AI-generated insights and buy/sell signals.

## 🌟 Features

### 📊 Real-Time Market Data Integration (NEW!)

- **Live Stock Prices** from Yahoo Finance
- **Options Chain Data** with actual market strikes and premiums
- **Implied Volatility (IV)** for each option contract
- **Open Interest (OI)** to gauge liquidity
- **Bid/Ask Spreads** for accurate pricing
- **Multiple Expiration Dates** - Choose from all available dates
- **Trading Volume** - See market activity
- **No API Key Required** - Free Yahoo Finance integration with **smart caching & retry logic**

### 🎯 Advanced Analytics (NEW!)

- **Greeks Calculation** - Delta, Gamma, Theta, Vega, Rho for entire strategies
- **Probability of Profit (POP)** - Statistical analysis of success likelihood
- **IV Rank & IV Percentile** - Determine if options are expensive or cheap
- **Expected Move** - Calculate anticipated price ranges (68% & 95% confidence)
- **Max Pain Calculator** - Identify price where option holders lose most
- **Put/Call Ratio** - Market sentiment analysis by volume and open interest
- **Historical IV Tracking** - Compare current vs historical volatility
- **Professional-Grade Analytics** - Tools used by institutional traders

### Options Strategies Supported

#### Basic Strategies

- **Long Call (bullish)** - Profit from upward price movement
- **Long Put (bearish)** - Profit from downward price movement
- **Covered Call** - Generate income on stock holdings
- **Cash Secured Put** - Acquire stock at discount or earn premium
- **Naked Call (bearish)** - High-risk bearish bet
- **Naked Put (bullish)** - High-risk bullish bet

#### Spread Strategies

- **Bull Call Spread** - Limited risk bullish strategy
- **Bear Put Spread** - Limited risk bearish strategy
- **Credit Spread** - Collect premium with defined risk
- **Calendar Spread** - Profit from time decay
- **Ratio Back Spread** - Volatile market strategy

#### Advanced Strategies

- **Iron Condor** - Profit from low volatility
- **Butterfly** - Profit from minimal price movement
- **Collar** - Protect stock holdings
- **Straddle** - Profit from high volatility
- **Strangle** - Profit from large price swings
- **Diagonal Spread** - Time and directional play
- **Double Diagonal** - Advanced income strategy

#### Custom Strategies

- Build custom strategies with 2-8 option legs
- Full control over strikes, premiums, and quantities

### AI-Powered Analysis

The app integrates **Google Gemini AI** to provide:

- 📊 Strategy overview and explanation
- 🎯 Market outlook and conditions
- ⚠️ Risk analysis and management
- 📈 Probability assessment
- 💡 **BUY/SELL/HOLD signals** with detailed rationale
- 🎚️ Key price levels to monitor
- 🚪 Exit strategy recommendations

### Visualizations

- **Interactive Payoff Diagrams** - See profit/loss at expiration
- **Real-time Metrics** - Max profit, max loss, breakevens
- **Risk/Reward Analysis** - Understand your risk exposure
- **Beautiful Dark Theme** - Modern, premium UI design

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Google API Key (for AI features)

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure Google API Key:**

   - Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

1. View the interactive payoff diagram
1. Review key metrics (max profit, max loss, breakeven points)
1. Click "Generate Comprehensive Report" for AI analysis
1. Review the AI-generated buy/sell signal and recommendations

## 🎨 Features Highlights

### Interactive Payoff Diagrams

- Real-time visualization of profit/loss at expiration
- Color-coded profit (green) and loss (red) zones
- Current stock price indicator
- Breakeven lines clearly marked

### Key Metrics Dashboard

- **Max Profit** - Maximum potential gain
- **Max Loss** - Maximum potential loss
- **Net Premium** - Initial cost or credit
- **Risk/Reward Ratio** - Profit potential vs risk
- **Breakeven Points** - Prices where strategy breaks even

### AI Analysis Report

The AI generates a comprehensive report including:

- Strategy explanation for beginners
- Ideal market conditions
- Risk factors and mitigation strategies
- Probability of success
- Clear BUY/SELL/HOLD recommendation
- Price levels to watch
- When to exit for profit or cut losses

## 🌐 Deploying to Streamlit Cloud

1. **Push your code to GitHub**

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Deploy your app:**

   - Connect your GitHub repository
   - Select the branch and file (`app.py`)
   - Add your `GOOGLE_API_KEY` in the Secrets section

4. **Your app will be live at:** `https://your-app-name.streamlit.app`

## ⚙️ Configuration

### Environment Variables

- `GOOGLE_API_KEY` - Your Google Gemini API key (required for AI features)

### Streamlit Configuration

The app uses custom configuration in `.streamlit/config.toml`:

- Premium dark theme
- Custom color scheme
- Optimized for performance

## 🛡️ Disclaimer

**⚠️ IMPORTANT RISK DISCLOSURE**

This tool is for **educational and informational purposes only**. Options trading involves substantial risk and is not suitable for all investors.

- Past performance does not guarantee future results
- AI-generated signals are not financial advice
- Always perform your own due diligence
- Consult with a licensed financial advisor before trading
- Never invest money you cannot afford to lose

The creators of this application are not responsible for any financial losses incurred from using this tool.

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📧 Support

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- AI powered by [Google Gemini](https://deepmind.google/technologies/gemini/)
- Charts created with [Plotly](https://plotly.com)
- Inspired by [OptionsProfitCalculator.com](https://www.optionsprofitcalculator.com/)

---

**Happy Trading! 📈**

Remember: The best investment you can make is in your education. Use this tool to learn and understand options strategies before risking real capital.
