# 🚀 Quick Start Guide

## First Time Setup (5 minutes)

### 1. Get Your Google API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key

### 2. Configure the App

1. Open `.streamlit/secrets.toml`
2. Replace `your-api-key-here` with your actual API key:
   ```toml
   GOOGLE_API_KEY = "AIzaSyC..."  # Your actual key
   ```
3. Save the file

### 3. Run the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8502`

## Using the App

### Basic Usage

1. **Select Strategy Category** (left sidebar)

   - Choose: Basic, Spreads, Advanced, or Custom

2. **Pick a Strategy**

   - Example: "Long Call" for bullish play
   - Example: "Iron Condor" for range-bound market

3. **Set Stock Price**

   - Enter current price of the underlying stock

4. **View Analysis**
   - Payoff diagram shows automatically
   - See Max Profit, Max Loss, Breakevens
   - Click "Generate Report" for AI analysis

### Example Scenarios

#### Scenario 1: Bullish on AAPL

```
Category: Basic
Strategy: Long Call
Stock Price: $180
- Buy 1 CALL @ $190, Premium $5
```

#### Scenario 2: Want Income from TSLA

```
Category: Basic
Strategy: Covered Call
Stock Price: $240
- Check "Include Stock Position"
- Shares: 100
- Sell 1 CALL @ $260, Premium $8
```

#### Scenario 3: Expect Low Volatility in SPY

```
Category: Advanced
Strategy: Iron Condor
Stock Price: $450
- Pre-configured with 4 legs
- Adjust strikes as needed
```

### Understanding the Charts

**Payoff Diagram:**

- **Green area** = Profit zone
- **Red area** = Loss zone
- **Yellow line** = Current stock price
- **Crossing points** = Breakeven prices

**Metrics Cards:**

- **Max Profit** = Best case scenario
- **Max Loss** = Worst case scenario
- **Net Premium** = Initial cost (negative) or credit (positive)
- **Risk/Reward** = Higher is better

### AI Analysis

Click **"Generate Comprehensive Report"** to get:

✅ Strategy explanation
✅ Market conditions needed
✅ Risk factors
✅ Probability of success
✅ **BUY/SELL/HOLD signal**
✅ Key price levels
✅ Exit strategy

**Signals:**

- 🟢 **BUY** = Good setup, consider entering
- 🔴 **SELL** = Poor risk/reward, avoid
- 🟡 **HOLD** = Wait for better entry

## Tips for Best Results

### 1. Realistic Premiums

Use actual market premiums for accurate analysis:

- Check real option chains
- Consider implied volatility
- Adjust for time to expiration

### 2. Multiple Scenarios

Test different configurations:

- Various strike prices
- Different expirations
- Multiple quantities

### 3. Risk Management

Always know your risks:

- Set max loss you can afford
- Use stop-loss orders
- Don't over-leverage

### 4. AI Insights

Use AI analysis to:

- Validate your thesis
- Identify risks you missed
- Plan entry/exit points
- Understand strategy mechanics

## Common Strategies Quick Reference

| Strategy         | Market View        | Risk      | Reward    | Best For           |
| ---------------- | ------------------ | --------- | --------- | ------------------ |
| Long Call        | Bullish            | Limited   | Unlimited | Strong uptrend     |
| Long Put         | Bearish            | Limited   | High      | Strong downtrend   |
| Covered Call     | Neutral-Bullish    | Unlimited | Limited   | Income generation  |
| Iron Condor      | Neutral            | Limited   | Limited   | Range-bound        |
| Straddle         | High Volatility    | High      | High      | Earnings/events    |
| Bull Call Spread | Moderately Bullish | Limited   | Limited   | Lower cost bullish |

## Keyboard Shortcuts

- `Ctrl + R` = Refresh page
- `Ctrl + Shift + R` = Hard refresh
- Adjust numbers with **arrow keys** after clicking input

## Troubleshooting

**Problem:** AI analysis not working

- ✅ Check API key in secrets.toml
- ✅ Restart the app
- ✅ Verify internet connection

**Problem:** Chart not updating

- ✅ Change any input value
- ✅ Refresh the page

**Problem:** App won't start

- ✅ Run: `pip install -r requirements.txt`
- ✅ Check Python version (3.8+)
- ✅ Try different port: `streamlit run app.py --server.port 8503`

## Next Steps

1. ✅ Experiment with different strategies
2. ✅ Compare multiple setups side-by-side
3. ✅ Save screenshots of promising strategies
4. ✅ Deploy to Streamlit Cloud (see DEPLOYMENT.md)
5. ✅ Share with trading community

## Important Reminder

⚠️ **This is an educational tool**

- Paper trade first
- Start small with real money
- Never risk more than you can lose
- Consult a financial advisor

---

**Ready to analyze options like a pro? Start exploring! 📈**

For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)
