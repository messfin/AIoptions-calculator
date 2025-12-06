# 📚 Options Strategies Reference Guide

This guide provides detailed information about all supported strategies, their use cases, and example configurations.

## 🎯 Basic Strategies

### 1. Long Call (Bullish)

**When to use:** Strong bullish outlook on underlying stock

**Structure:**

- Buy 1 Call Option

**Example Setup:**

- Stock Price: $100
- Buy 1 CALL @ $105
- Premium Paid: $5
- Expiration: 30 days

**Profit/Loss:**

- Max Profit: Unlimited
- Max Loss: Premium paid ($500 for 1 contract)
- Breakeven: Strike + Premium ($110)

**Best Scenarios:**

- Strong uptrend expected
- Before positive catalysts (earnings, product launch)
- Limited downside risk needed

---

### 2. Long Put (Bearish)

**When to use:** Strong bearish outlook on underlying stock

**Structure:**

- Buy 1 Put Option

**Example Setup:**

- Stock Price: $100
- Buy 1 PUT @ $95
- Premium Paid: $5
- Expiration: 30 days

**Profit/Loss:**

- Max Profit: Strike - Premium ($9,000 if stock goes to $0)
- Max Loss: Premium paid ($500)
- Breakeven: Strike - Premium ($90)

**Best Scenarios:**

- Strong downtrend expected
- Hedging long stock positions
- Before negative catalysts

---

### 3. Covered Call

**When to use:** Own stock, want income, neutral-to-slightly-bullish

**Structure:**

- Own 100 shares of stock
- Sell 1 Call Option

**Example Setup:**

- Stock Price: $100 (own 100 shares)
- Sell 1 CALL @ $110
- Premium Collected: $4
- Expiration: 30 days

**Profit/Loss:**

- Max Profit: (Strike - Stock Price) + Premium ($14/share = $1,400)
- Max Loss: Stock can go to $0 (offset by premium)
- Breakeven: Stock Purchase Price - Premium

**Best Scenarios:**

- Generate income on holdings
- Willing to sell at strike price
- Sideways or slowly rising market

---

### 4. Cash Secured Put

**When to use:** Want to buy stock at lower price or collect premium

**Structure:**

- Sell 1 Put Option
- Have cash to buy 100 shares at strike

**Example Setup:**

- Stock Price: $100
- Sell 1 PUT @ $95
- Premium Collected: $4
- Cash Reserved: $9,500

**Profit/Loss:**

- Max Profit: Premium collected ($400)
- Max Loss: Strike - Premium ($9,100 if stock goes to $0)
- Breakeven: Strike - Premium ($91)

**Best Scenarios:**

- Want to own stock at discount
- Bullish long-term outlook
- Okay with obligation to buy

---

## 📊 Spread Strategies

### 5. Bull Call Spread

**When to use:** Moderately bullish, want to reduce cost

**Structure:**

- Buy 1 Call (lower strike)
- Sell 1 Call (higher strike)

**Example Setup:**

- Stock Price: $100
- Buy 1 CALL @ $100, Premium $5
- Sell 1 CALL @ $110, Premium $2
- Net Cost: $3

**Profit/Loss:**

- Max Profit: (Strikes Difference) - Net Cost ($7/share = $700)
- Max Loss: Net Cost ($300)
- Breakeven: Lower Strike + Net Cost ($103)

**Best Scenarios:**

- Moderate bullish outlook
- Lower cost than long call
- Defined risk and reward

---

### 6. Bear Put Spread

**When to use:** Moderately bearish, want to reduce cost

**Structure:**

- Buy 1 Put (higher strike)
- Sell 1 Put (lower strike)

**Example Setup:**

- Stock Price: $100
- Buy 1 PUT @ $100, Premium $5
- Sell 1 PUT @ $90, Premium $2
- Net Cost: $3

**Profit/Loss:**

- Max Profit: (Strikes Difference) - Net Cost ($7/share = $700)
- Max Loss: Net Cost ($300)
- Breakeven: Higher Strike - Net Cost ($97)

**Best Scenarios:**

- Moderate bearish outlook
- Lower cost than long put
- Defined risk and reward

---

### 7. Calendar Spread (Time Spread)

**When to use:** Profit from time decay, neutral outlook

**Structure:**

- Sell near-term option
- Buy longer-term option (same strike)

**Example Setup:**

- Stock Price: $100
- Sell 1 CALL @ $100, 30 days, Premium $5
- Buy 1 CALL @ $100, 90 days, Premium $8
- Net Cost: $3

**Profit/Loss:**

- Max Profit: Varies with time decay
- Max Loss: Net Cost ($300)
- Works best if stock stays near strike

**Best Scenarios:**

- Expect low volatility
- Stock near strike at near-term expiration
- Advanced strategy

---

## 🎓 Advanced Strategies

### 8. Iron Condor

**When to use:** Expect low volatility, range-bound stock

**Structure:**

- Sell OTM Put
- Buy further OTM Put
- Sell OTM Call
- Buy further OTM Call

**Example Setup:**

- Stock Price: $100
- Buy 1 PUT @ $90, Premium $1
- Sell 1 PUT @ $95, Premium $3
- Sell 1 CALL @ $105, Premium $3
- Buy 1 CALL @ $110, Premium $1
- Net Credit: $4

**Profit/Loss:**

- Max Profit: Net Credit ($400)
- Max Loss: (Widest Spread) - Net Credit ($600)
- Two Breakevens: $91 and $109

**Best Scenarios:**

- Low volatility expected
- Stock range-bound
- High probability income

---

### 9. Butterfly

**When to use:** Expect stock to stay exactly at middle strike

**Structure:**

- Buy 1 Call (lower strike)
- Sell 2 Calls (middle strike)
- Buy 1 Call (higher strike)

**Example Setup:**

- Stock Price: $100
- Buy 1 CALL @ $95, Premium $8
- Sell 2 CALLS @ $100, Premium $5 each
- Buy 1 CALL @ $105, Premium $3
- Net Cost: $1

**Profit/Loss:**

- Max Profit: (Strike Width) - Net Cost ($400 if stock at $100)
- Max Loss: Net Cost ($100)
- Two Breakevens: ~$96 and ~$104

**Best Scenarios:**

- Strong conviction on exact price
- Low volatility expected
- Excellent risk/reward ratio

---

### 10. Straddle

**When to use:** Expect large price movement (either direction)

**Structure:**

- Buy 1 Call (ATM)
- Buy 1 Put (ATM)

**Example Setup:**

- Stock Price: $100
- Buy 1 CALL @ $100, Premium $5
- Buy 1 PUT @ $100, Premium $5
- Total Cost: $10

**Profit/Loss:**

- Max Profit: Unlimited (either direction)
- Max Loss: Total Premium ($1,000)
- Two Breakevens: $90 and $110

**Best Scenarios:**

- Before earnings announcements
- Expecting volatility spike
- Uncertain direction but big move expected

---

### 11. Strangle

**When to use:** Like straddle but lower cost

**Structure:**

- Buy 1 OTM Call
- Buy 1 OTM Put

**Example Setup:**

- Stock Price: $100
- Buy 1 CALL @ $105, Premium $3
- Buy 1 PUT @ $95, Premium $3
- Total Cost: $6

**Profit/Loss:**

- Max Profit: Unlimited (either direction)
- Max Loss: Total Premium ($600)
- Two Breakevens: $89 and $111

**Best Scenarios:**

- Before major events
- Lower cost than straddle
- Need bigger move to profit

---

### 12. Collar

**When to use:** Protect long stock position

**Structure:**

- Own 100 shares
- Buy 1 OTM Put (protection)
- Sell 1 OTM Call (income)

**Example Setup:**

- Stock Price: $100 (own 100 shares)
- Buy 1 PUT @ $95, Premium $3
- Sell 1 CALL @ $110, Premium $4
- Net Credit: $1

**Profit/Loss:**

- Max Profit: (Call Strike - Stock Price) + Net ($11/share)
- Max Loss: (Stock Price - Put Strike) - Net ($4/share)
- Protected below $95, capped above $110

**Best Scenarios:**

- Earnings protection
- Lock in profits
- Free or low-cost insurance

---

## 🎨 Custom Strategy Ideas

### Aggressive Bull (Custom 3 Legs)

- Buy 2 CALLS @ $105
- Sell 1 CALL @ $110
- Buy 10 shares of stock

### Income Generator (Custom 4 Legs)

- Own 200 shares
- Sell 1 CALL @ $110
- Sell 1 CALL @ $115
- Sell 1 PUT @ $90

### Volatility Play (Custom 5 Legs)

- Buy 1 CALL @ $100
- Buy 1 CALL @ $105
- Buy 1 PUT @ $100
- Buy 1 PUT @ $95
- Sell 2 CALLS @ $110

---

## 📊 Strategy Selection Guide

### By Market Outlook

| Outlook              | Strategies                           |
| -------------------- | ------------------------------------ |
| **Strong Bullish**   | Long Call, Bull Call Spread          |
| **Moderate Bullish** | Bull Call Spread, Covered Call       |
| **Strong Bearish**   | Long Put, Bear Put Spread            |
| **Moderate Bearish** | Bear Put Spread, Naked Call          |
| **Neutral**          | Iron Condor, Butterfly, Covered Call |
| **High Volatility**  | Straddle, Strangle                   |
| **Low Volatility**   | Iron Condor, Calendar Spread         |

### By Risk Tolerance

| Risk Level      | Strategies                              |
| --------------- | --------------------------------------- |
| **Low Risk**    | Covered Call, Cash Secured Put, Spreads |
| **Medium Risk** | Long Calls/Puts, Iron Condor, Butterfly |
| **High Risk**   | Naked Options, Straddle, Strangle       |

### By Time Horizon

| Timeframe                    | Strategies                    |
| ---------------------------- | ----------------------------- |
| **Short-term (0-30 days)**   | Straddle, Strangle, Butterfly |
| **Medium-term (30-90 days)** | Most spreads, Long options    |
| **Long-term (90+ days)**     | LEAPS, Calendar Spreads       |

---

## 💡 Pro Tips

### 1. Volatility Considerations

- **High IV:** Sell premium (covered calls, iron condors)
- **Low IV:** Buy options (long calls/puts, straddles)

### 2. Greeks to Watch

- **Delta:** Direction sensitivity
- **Theta:** Time decay
- **Vega:** Volatility sensitivity
- **Gamma:** Delta change rate

### 3. Risk Management

- Never risk more than 2-5% per trade
- Use stop losses
- Size positions appropriately
- Diversify strategies

### 4. Timing

- **Before Earnings:** Straddles, Strangles
- **After Earnings:** Calendar Spreads
- **Bull Markets:** Call strategies
- **Bear Markets:** Put strategies

---

## 📚 Resources for Learning

1. **Options Playbook** - Visual guide to strategies
2. **CBOE Education** - Official options exchange
3. **Tastytrade** - Free options education
4. **OptionAlpha** - Strategy examples
5. **Investopedia** - Options basics

---

## ⚠️ Risk Warnings

- Options can expire worthless
- Naked options have unlimited risk
- Spreads limit gains and losses
- Time decay works against buyers
- Volatility can be unpredictable

**Always paper trade first!**

---

_This reference guide is for educational purposes only. Always do your own research and consult with financial professionals before trading._
