# 📘 ZMtech AI Options Calculator Pro - Basic Usage Guide

## Welcome! 👋

This guide will walk you through using the options calculator in **5 simple steps**. No prior options experience needed!

---

## 🚀 Getting Started

### Step 1: Launch the App

**Visit the live app:**

- Open your browser and go to: **https://zmtech.streamlit.app/**
- No installation needed!

**Or run locally:**

1. Open your terminal/command prompt
2. Navigate to the options folder: `cd d:\options`
3. Run: `streamlit run app.py`
4. The app opens automatically in your browser at `http://localhost:8502`

---

## 📖 Step-by-Step Tutorial

### Step 2: Choose Your Strategy

**On the left sidebar:**

1. **Select Category:**

   - **Basic** - Simple strategies (Long Call, Long Put, etc.)
   - **Spreads** - Two-leg strategies (Bull Call Spread, etc.)
   - **Advanced** - Complex strategies (Iron Condor, etc.)
   - **Custom** - Build your own (2-8 legs)

2. **Select Strategy:**
   - For beginners, start with **"Long Call"** (bullish bet)
   - Or **"Long Put"** (bearish bet)

**Example:** Let's choose **Basic → Long Call**

---

### Step 3: Enter Stock Information

**Two ways to do this:**

#### Option A: Manual Entry (Simpler)

1. Leave **"Use Real-Time Market Data"** unchecked
2. Enter stock price manually (e.g., `100`)
3. Click in the field and type the number

#### Option B: Real-Time Data (Recommended)

1. ✅ Check **"Use Real-Time Market Data"**
2. **Enter stock symbol** (e.g., `AAPL` for Apple)
3. Wait 2-3 seconds for data to load
4. **Select expiration date** from dropdown
5. Current stock price loads automatically! 📈

**For this tutorial, use AAPL with real-time data.**

---

### Step 4: Review Your Strategy

**In the main area, you'll see:**

#### Left Side - Strategy Configuration

- Shows the option leg(s) automatically configured
- For **Long Call**, you'll see:
  - **BUY 1 CALL**
  - Strike price (e.g., $189 if AAPL is at $180)
  - Premium (cost per share)
  - Expiration (30 days default)

#### Right Side - Key Metrics

Look for these important numbers:

📊 **Max Profit:** How much you can make (often "Unlimited" for long calls)

📉 **Max Loss:** How much you can lose (the premium you paid)

⚖️ **Risk/Reward:** Ratio of profit potential to risk

🎯 **Breakeven:** Price stock needs to reach to break even

**Example interpretation:**

```
Max Profit: Unlimited
Max Loss: -$500 (the premium you paid)
Breakeven: $185

This means:
- You paid $500 for the option
- If stock goes above $185, you profit
- If below $185 at expiration, you lose your $500
```

---

### Step 5: View the Payoff Diagram

**The chart in the middle shows:**

- **Green area** = Profit zone 💰
- **Red area** = Loss zone 📉
- **Yellow dotted line** = Current stock price
- **White line** = Your profit/loss at different prices

**How to read it:**

1. Find the current stock price (yellow line)
2. Look where the blue profit line crosses zero (breakeven)
3. See how profit grows as stock price increases (for long call)

**Hover over the chart** to see exact profit/loss at any price!

---

## 🎯 Advanced Analytics (Optional)

Scroll down on the right sidebar to see:

### Greeks (Always Shown)

- **Delta:** How much option price changes with stock
- **Theta:** How much you lose per day (time decay)
- **Other Greeks:** Gamma, Vega, Rho

### Probability of Profit

- Shows your % chance of making money
- Higher % = safer trade

**Example:**

```
POP: 45%
→ This trade has 45% chance of profit
→ Lower probability, but potentially high reward
```

### When Using Real-Time Data

You also get:

- **IV Rank** - Is IV high or low?
- **Expected Move** - How far might stock move?
- **Max Pain** - Where might price gravitate?
- **Put/Call Ratio** - Market sentiment

---

## 💡 Common Strategies Explained

### 1. Long Call (Bullish)

**When to use:** You think stock will go UP

**Setup:**

- Buy 1 Call option
- Choose strike above current price

**Example:**

```
Stock: $100
Buy CALL @ $105 for $3

Profit if: Stock above $108 at expiration
Max Loss: $300 (premium paid)
Max Profit: Unlimited
```

---

### 2. Long Put (Bearish)

**When to use:** You think stock will go DOWN

**Setup:**

- Buy 1 Put option
- Choose strike below current price

**Example:**

```
Stock: $100
Buy PUT @ $95 for $3

Profit if: Stock below $92 at expiration
Max Loss: $300 (premium paid)
Max Profit: $9,500 (if stock goes to $0)
```

---

### 3. Bull Call Spread (Moderate Bullish)

**When to use:** You think stock will go up, but want cheaper entry

**Setup:**

- Buy 1 Call (lower strike)
- Sell 1 Call (higher strike)

**Example:**

```
Stock: $100
Buy CALL @ $100 for $5
Sell CALL @ $110 for $2
Net Cost: $3

Max Profit: $700 (strike difference - cost)
Max Loss: $300 (net cost)
Breakeven: $103
```

**Why it's better than Long Call:**

- Cheaper entry ($3 vs $5)
- Defined max profit and loss
- Higher probability of profit

---

### 4. Iron Condor (Neutral/Income)

**When to use:** You think stock will stay flat

**Setup:**

- 4 options (pre-configured in app)
- Profit if stock stays in range

**Example:**

```
Stock: $100
Strategy collects $400 premium

Profit Range: $95 - $105
Max Profit: $400 (if stock stays in range)
Max Loss: $600 (if moves outside range)
POP: 70%+
```

**Perfect for:** Generating income in sideways markets

---

## 🔧 Customizing Your Strategy

### To Modify Pre-Configured Strategies:

1. After selecting a strategy, check **"Customize This Strategy"**
2. Adjust parameters:
   - **Strike Price** - Choose different strike
   - **Premium** - Enter actual market premium
   - **Quantity** - Number of contracts
3. Watch payoff diagram update in real-time!

### Creating Custom Strategies:

1. Select **Custom** category
2. Choose number of legs (2, 3, 4, etc.)
3. For each leg, select:
   - **Type:** Call or Put
   - **Action:** Buy or Sell
   - **Strike:** Price level
   - **Premium:** Cost per share
   - **Quantity:** Number of contracts
4. Build complex strategies like Iron Butterfly, etc.

---

## 🤖 Getting AI Analysis

**After setting up your strategy:**

1. Scroll to bottom of page
2. Click **"🚀 Generate Comprehensive Report"**
3. Wait 5-10 seconds
4. Read AI analysis including:
   - Strategy explanation
   - Market conditions needed
   - Risk analysis
   - **BUY/SELL/HOLD recommendation**
   - Exit strategy

**Note:** Requires Google API key (see setup guide)

---

## 📱 Real-World Example Walkthrough

### Trading Apple (AAPL) - Bull Call Spread

**Scenario:** You think AAPL will go from $180 to $190 in 30 days.

**Steps:**

1. **Select Strategy:**

   - Category: Spreads
   - Strategy: Bull Call Spread

2. **Enter Symbol:**

   - ✅ Check "Use Real-Time Market Data"
   - Symbol: AAPL
   - Expiration: Choose 30-day option

3. **Review Pre-Configured Setup:**

   ```
   Buy CALL @ $183 for $6.50
   Sell CALL @ $193 for $2.50
   Net Cost: $4.00 per share
   Total Cost: $400 per contract
   ```

4. **Check Metrics:**

   ```
   Max Profit: $600
   Max Loss: $400
   Risk/Reward: 1.5
   Breakeven: $187
   POP: 45%
   ```

5. **Analyze Chart:**

   - Green above $187 ✅
   - Max profit at $193+
   - Looks good if bullish!

6. **Review Greeks:**

   ```
   Delta: +35 (profitable if stock rises)
   Theta: -$8/day (losing to time decay)
   Vega: +$25 (benefits from IV increase)
   ```

7. **Check Advanced Analytics:**

   ```
   IV Rank: 45 (Neutral)
   Expected Move: ±$8 (stock could hit $188)
   Max Pain: $185 (stock may gravitate here)
   P/C Ratio: 0.88 (Slightly bullish)
   ```

8. **Get AI Opinion:**

   - Click "Generate Report"
   - AI says: "HOLD - Neutral setup, consider if strongly bullish"

9. **Make Decision:**
   - Moderately bullish on AAPL? ✅ Execute
   - Very bullish? Consider Long Call instead
   - Uncertain? Wait for better setup

---

## ⚠️ Important Tips for Beginners

### 1. **Start Small**

- Paper trade first
- Use 1 contract to learn
- Don't risk more than you can afford to lose

### 2. **Understand Greeks**

- **Positive Theta** = You make money daily (sellers)
- **Negative Theta** = You lose money daily (buyers)
- **High Delta** = More directional exposure

### 3. **Check Probability of Profit**

- **POP > 70%** = High probability, small profit
- **POP < 40%** = Low probability, large profit
- Balance your portfolio!

### 4. **Use Real-Time Data**

- Always verify premiums before trading
- Check Open Interest (liquidity)
- Review actual IV levels

### 5. **Set Profit Targets**

- Example: Close at 50% max profit
- Don't be greedy
- Protect your gains

### 6. **Manage Risk**

- Know your max loss before entering
- Set stop losses
- Don't over-leverage

---

## 🎓 Learning Path

### Week 1: Basics

- ✅ Learn Long Call and Long Put
- ✅ Understand payoff diagrams
- ✅ Study Delta and Theta
- ✅ Paper trade 5 setups

### Week 2: Spreads

- ✅ Try Bull/Bear Call Spreads
- ✅ Learn about defined risk
- ✅ Compare to long options
- ✅ Paper trade spreads

### Week 3: Income

- ✅ Study Iron Condor
- ✅ Sell premium strategies
- ✅ Understand high POP trades
- ✅ Learn to roll positions

### Week 4: Advanced

- ✅ Calendar spreads
- ✅ Diagonal spreads
- ✅ Custom strategies
- ✅ Live trade small

---

## 🆘 Troubleshooting

### "No data loading for my symbol"

- Check spelling (use ticker symbols)
- Try liquid stocks first (AAPL, TSLA, SPY)
- Ensure internet connection

### "Can't see Greeks/Analytics"

- Scroll down in right sidebar
- May be in collapsed expander
- Should always show for any strategy

### "AI analysis not working"

- Need to configure Google API key
- See QUICKSTART.md for setup
- Free API key from Google AI Studio

### "Payoff diagram looks wrong"

- Verify all strikes are correct
- Check buy/sell for each leg
- Ensure quantities are right
- Try refreshing page

---

## 📚 What to Read Next

**For Basic Understanding:**

- 📖 QUICKSTART.md - 5-minute quick start

**For Real-Time Data:**

- 📊 REALTIME_DATA_GUIDE.md - Using Yahoo Finance data

**For Strategy Details:**

- 📈 STRATEGIES_GUIDE.md - All strategy explanations
- 🆕 NEW_STRATEGIES_GUIDE.md - Newly added strategies

**For Advanced Features:**

- 🎯 ADVANCED_ANALYTICS_GUIDE.md - Greeks, IV, POP explained

**For Deployment:**

- 🚀 DEPLOYMENT.md - Deploy to cloud

---

## ✅ Quick Checklist

**Before Your First Trade:**

- [ ] Understand the strategy
- [ ] Know your max loss
- [ ] Check breakeven price
- [ ] Verify probability of profit
- [ ] Review Greeks (Delta, Theta)
- [ ] Check IV conditions
- [ ] Set profit target
- [ ] Set stop loss
- [ ] Start with 1 contract
- [ ] Paper trade first!

---

## 🎯 Summary

**Basic Workflow:**

```
1. Choose Strategy →
2. Enter Stock Symbol →
3. Review Setup →
4. Check Metrics →
5. Analyze Chart →
6. Get AI Opinion →
7. Make Decision
```

**Remember:**

- 📊 Green = Profit, Red = Loss
- 🎲 Higher POP = Safer but smaller gains
- ⏰ Theta = Time decay (friend or enemy)
- 📈 Delta = Directional exposure
- 💡 Start simple, learn gradually

**You're now ready to analyze options like a pro!** 🚀

---

## 🤝 Need Help?

- Check other guides in the `d:\options` folder
- All guides are in Markdown (.md) format
- Read STRATEGIES_GUIDE.md for strategy deep-dives
- Read ADVANCED_ANALYTICS_GUIDE.md for Greeks

**Happy Trading! 📈💰**

_Remember: This is for educational purposes. Always consult a financial advisor and never invest more than you can afford to lose._
