# 📚 New Strategy Templates - Quick Reference

## ✅ Strategies Added

All requested option strategies now have pre-configured templates!

---

## 📊 Spread Strategies

### 1. **Credit Spread** (Bull Put Credit Spread)

**Type:** Income Strategy  
**Market View:** Neutral to Bullish  
**Risk/Reward:** Limited/Limited

**Structure:**

- SELL 1 PUT @ 98% of stock price (Premium: 3%)
- BUY 1 PUT @ 93% of stock price (Premium: 1%)

**Net Credit:** 2% of stock price

**How it works:**

- Collect premium upfront (credit)
- Profit if stock stays above short put strike
- Max profit = Net credit received
- Max loss = Strike width - Net credit

**Best for:**

- Bullish or sideways market
- Want to collect premium
- Defined risk income strategy

---

### 2. **Call Spread** (Debit Call Spread)

**Type:** Directional Bullish  
**Market View:** Moderately Bullish  
**Risk/Reward:** Limited/Limited

**Structure:**

- BUY 1 CALL @ 102% of stock price (Premium: 4%)
- SELL 1 CALL @ 108% of stock price (Premium: 2%)

**Net Debit:** 2% of stock price

**How it works:**

- Pay net debit upfront
- Profit from upward movement
- Max profit = Strike width - Net debit
- Max loss = Net debit paid

**Best for:**

- Moderately bullish outlook
- Lower cost than long call
- Defined risk directional trade

---

### 3. **Put Spread** (Debit Put Spread)

**Type:** Directional Bearish  
**Market View:** Moderately Bearish  
**Risk/Reward:** Limited/Limited

**Structure:**

- BUY 1 PUT @ 98% of stock price (Premium: 4%)
- SELL 1 PUT @ 92% of stock price (Premium: 2%)

**Net Debit:** 2% of stock price

**How it works:**

- Pay net debit upfront
- Profit from downward movement
- Max profit = Strike width - Net debit
- Max loss = Net debit paid

**Best for:**

- Moderately bearish outlook
- Lower cost than long put
- Defined risk directional trade

---

### 4. **Calendar Spread** (Time Spread)

**Type:** Neutral/Volatility  
**Market View:** Neutral with volatility expectation  
**Risk/Reward:** Limited/Limited

**Structure:**

- SELL 1 CALL @ ATM, 20 days (Premium: 4%) - Near-term
- BUY 1 CALL @ ATM, 60 days (Premium: 6%) - Far-term

**Net Debit:** 2% of stock price

**How it works:**

- Near-term option decays faster (Theta play)
- Profit if stock stays near strike at near-term expiration
- Then hold long-term option for potential upside
- Max profit occurs at strike at near expiration

**Best for:**

- Stock likely to stay flat short-term
- Expecting volatility increase
- Advanced time decay strategy

---

### 5. **Ratio Back Spread**

**Type:** Volatile/Unlimited Upside  
**Market View:** Expect BIG move up  
**Risk/Reward:** Limited/Unlimited

**Structure:**

- SELL 1 CALL @ ATM (Premium: 5%)
- BUY 2 CALLS @ 110% of stock price (Premium: 2% each)

**Net Credit/Debit:** 1% credit (in this example)

**How it works:**

- Profit from large upward moves
- Neutral to small profit if flat
- Small loss zone between strikes
- Unlimited profit potential above breakeven

**Breakevens:**

- Lower: Near ATM strike + net debit
- Upper: Typically well above higher strike

**Best for:**

- Expecting explosive upside move
- Earnings plays or catalysts
- Wanting unlimited profit potential
- Okay with defined risk zone

---

## 🎓 Advanced Strategies

### 6. **Iron Condor** ✅ (Already Available)

**Quick Summary:** Sell both call and put spreads for income. Profit if stock stays in range.

---

### 7. **Butterfly** ✅ (Already Available)

**Quick Summary:** High probability, low-cost bet on specific price. Max profit if stock at middle strike.

---

### 8. **Diagonal Spread**

**Type:** Hybrid Directional/Time  
**Market View:** Moderately Bullish with time  
**Risk/Reward:** Limited/Limited

**Structure:**

- SELL 1 CALL @ 105% of stock price, 30 days (Premium: 4%)
- BUY 1 CALL @ 110% of stock price, 60 days (Premium: 3%)

**Net Debit:** 1% (varies based on time/strikes)

**How it works:**

- Like calendar spread but different strikes
- Near-term decays while holding longer-term
- Can roll short leg monthly for income
- Directional bias with time decay benefit

**Best for:**

- Long-term bullish, near-term neutral
- Monthly income from rolling
- Advanced traders

---

### 9. **Double Diagonal**

**Type:** Advanced Income  
**Market View:** Neutral/Range-bound  
**Risk/Reward:** Limited/Limited

**Structure:**

- SELL 1 PUT @ 95%, 30 days (Premium: 3%)
- BUY 1 PUT @ 90%, 60 days (Premium: 2%)
- SELL 1 CALL @ 105%, 30 days (Premium: 3%)
- BUY 1 CALL @ 110%, 60 days (Premium: 2%)

**Net Credit:** 2%

**How it works:**

- Like Iron Condor but with different expirations
- Reduces risk vs regular Iron Condor
- Can roll short legs for ongoing income
- Profit from time decay on short options

**Best for:**

- Range-bound market
- Monthly income generation
- Advanced traders comfortable rolling

---

## 📊 Strategy Comparison Table

| Strategy              | Legs | Direction    | Risk    | Reward    | Complexity |
| --------------------- | ---- | ------------ | ------- | --------- | ---------- |
| **Credit Spread**     | 2    | Bullish      | Limited | Limited   | ⭐⭐       |
| **Call Spread**       | 2    | Bullish      | Limited | Limited   | ⭐⭐       |
| **Put Spread**        | 2    | Bearish      | Limited | Limited   | ⭐⭐       |
| **Calendar Spread**   | 2    | Neutral      | Limited | Limited   | ⭐⭐⭐⭐   |
| **Ratio Back Spread** | 3    | Very Bullish | Limited | Unlimited | ⭐⭐⭐⭐   |
| **Iron Condor**       | 4    | Neutral      | Limited | Limited   | ⭐⭐⭐     |
| **Butterfly**         | 3    | Pinpoint     | Limited | High      | ⭐⭐⭐     |
| **Diagonal Spread**   | 2    | Bullish/Time | Limited | Limited   | ⭐⭐⭐⭐⭐ |
| **Double Diagonal**   | 4    | Neutral/Time | Limited | Limited   | ⭐⭐⭐⭐⭐ |

---

## 🎯 Quick Selection Guide

### **Want Income?**

- Credit Spread (bullish)
- Iron Condor (neutral)
- Double Diagonal (advanced)

### **Moderately Bullish?**

- Call Spread
- Bull Call Spread
- Diagonal Spread (with time)

### **Moderately Bearish?**

- Put Spread
- Bear Put Spread

### **Expect Flat Market?**

- Calendar Spread
- Iron Condor
- Butterfly (at specific price)

### **Expect BIG Move?**

- Ratio Back Spread (upside)
- Straddle (either direction)
- Strangle (either direction)

### **Want Advanced Income Strategy?**

- Calendar Spread
- Diagonal Spread
- Double Diagonal

---

## 💡 Pro Tips

### Credit Spread

- Sell 1 standard deviation out
- Target 1-2% return on risk
- 70-80% probability of profit

### Calendar Spread

- Works best in low volatility
- Profit if IV increases
- Close at near-term expiration

### Ratio Back Spread

- Can do for net credit or small debit
- Unlimited upside potential
- Best before earnings/catalysts

### Diagonal Spread

- Roll short leg monthly
- "Poor Man's Covered Call"
- Manage like calendar spread

### Double Diagonal

- Roll short options every 30-45 days
- Target 5-10% monthly return
- Requires active management

---

## ⚠️ Risk Warnings

**Calendar/Diagonal Spreads:**

- Require specific conditions to profit
- More complex to manage
- Time decay works against you if wrong

**Ratio Back Spreads:**

- Have defined loss zone
- Need big moves to profit
- Best for very experienced traders

**Double Diagonal:**

- Advanced strategy
- Requires understanding of rolling
- 4 legs = higher commissions

---

## 🎓 Learning Path

**Beginner → Intermediate → Advanced:**

1. Start: Call/Put Spreads ⭐⭐
2. Progress: Iron Condor, Butterfly ⭐⭐⭐
3. Master: Calendar/Diagonal ⭐⭐⭐⭐
4. Expert: Ratio Spreads, Double Diagonal ⭐⭐⭐⭐⭐

---

## 📚 All Available Strategies

### Basic (6):

- Long Call, Long Put
- Covered Call, Cash Secured Put
- Naked Call, Naked Put

### Spreads (7):

- ✅ **Bull Call Spread**
- ✅ **Bear Put Spread**
- ✅ **Credit Spread** (NEW!)
- ✅ **Call Spread** (NEW!)
- ✅ **Put Spread** (NEW!)
- ✅ **Calendar Spread** (NEW!)
- ✅ **Ratio Back Spread** (NEW!)

### Advanced (7):

- ✅ **Iron Condor**
- ✅ **Butterfly**
- Collar
- ✅ **Diagonal Spread** (NEW!)
- ✅ **Double Diagonal** (NEW!)
- Straddle
- Strangle

### Custom:

- 2, 3, 4, 5, 6, 8 Leg strategies

**Total: 20+ Pre-configured Strategies! 🎉**

---

All strategies are now available in the app with realistic pre-configured parameters. Just select from the dropdown and start analyzing!
