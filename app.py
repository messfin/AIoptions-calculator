import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import google.generativeai as genai
from dataclasses import dataclass
from typing import List, Dict, Literal, Optional
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import math
from report_generator import create_word_report, create_pdf_report

# Configure page
st.set_page_config(
    page_title="ZMtech AI Options Calculator Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        }
        
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 700;
        }
        
        .strategy-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            cursor: pointer;
            margin: 10px 0;
        }
        
        .strategy-card:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 20px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .ai-report {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 25px;
            border-left: 4px solid #667eea;
            margin: 20px 0;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 28px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
        
        .sidebar .sidebar-content {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 28px;
            font-weight: 700;
        }
        
        .buy-signal {
            color: #4ade80;
            font-weight: 700;
            font-size: 24px;
        }
        
        .sell-signal {
            color: #f87171;
            font-weight: 700;
            font-size: 24px;
        }
        
        .hold-signal {
            color: #fbbf24;
            font-weight: 700;
            font-size: 24px;
        }
        
        .stSelectbox, .stNumberInput, .stDateInput {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

@dataclass
class OptionLeg:
    """Represents a single option leg in a strategy"""
    type: Literal["call", "put"]
    action: Literal["buy", "sell"]
    strike: float
    premium: float
    quantity: int
    expiration_days: int

@dataclass
class OptionStrategy:
    """Represents a complete options trading strategy"""
    name: str
    legs: List[OptionLeg]
    stock_price: float
    underlying_shares: int = 0  # For strategies like covered call
    symbol: str = "UNKNOWN"
    
    def calculate_payoff(self, price_range: np.ndarray) -> np.ndarray:
        """Calculate total payoff across price range"""
        total_payoff = np.zeros_like(price_range)
        
        # Add stock position payoff if any
        if self.underlying_shares != 0:
            total_payoff += self.underlying_shares * (price_range - self.stock_price)
        
        # Calculate each option leg
        for leg in self.legs:
            leg_payoff = np.zeros_like(price_range)
            
            if leg.type == "call":
                intrinsic = np.maximum(price_range - leg.strike, 0)
            else:  # put
                intrinsic = np.maximum(leg.strike - price_range, 0)
            
            if leg.action == "buy":
                leg_payoff = (intrinsic - leg.premium) * leg.quantity
            else:  # sell
                leg_payoff = (leg.premium - intrinsic) * leg.quantity
            
            total_payoff += leg_payoff
        
        return total_payoff
    
    def get_metrics(self) -> Dict:
        """Calculate key strategy metrics"""
        price_range = np.linspace(self.stock_price * 0.5, self.stock_price * 1.5, 1000)
        payoff = self.calculate_payoff(price_range)
        
        max_profit = np.max(payoff)
        max_loss = np.min(payoff)
        
        # Find breakeven points (where payoff crosses zero)
        breakeven_points = []
        for i in range(len(payoff) - 1):
            if (payoff[i] <= 0 and payoff[i+1] > 0) or (payoff[i] >= 0 and payoff[i+1] < 0):
                breakeven_points.append(price_range[i])
        
        # Calculate net premium (initial cost/credit)
        net_premium = sum(
            leg.premium * leg.quantity * (-1 if leg.action == "buy" else 1)
            for leg in self.legs
        )
        
        return {
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven_points": breakeven_points,
            "net_premium": net_premium,
            "risk_reward_ratio": abs(max_profit / max_loss) if max_loss != 0 else float('inf')
        }

@st.cache_data(ttl=1800)  # Cache for 30 minutes (increased from 5)
def fetch_stock_data(symbol: str) -> Optional[Dict]:
    """Fetch real-time stock data from Yahoo Finance with retry logic"""
    import time
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Add headers to avoid rate limiting
            ticker = yf.Ticker(symbol)
            
            # Try to get history first (less likely to be rate limited)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                # If today's data not available, try 5 days
                hist = ticker.history(period="5d")
                
            if hist.empty:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # Try to get info, but don't fail if rate limited
            try:
                info = ticker.info
                company_name = info.get('longName', symbol)
                previous_close = info.get('previousClose', current_price)
                market_cap = info.get('marketCap', 0)
            except:
                # If info fails, use data from history
                company_name = symbol
                previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                market_cap = 0
            
            return {
                'symbol': symbol,
                'price': current_price,
                'previous_close': previous_close,
                'volume': hist['Volume'].iloc[-1],
                'market_cap': market_cap,
                'company_name': company_name,
                'change': current_price - previous_close,
                'change_percent': ((current_price - previous_close) / previous_close) * 100 if previous_close else 0
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if 'rate' in error_msg or '429' in error_msg or 'too many' in error_msg:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                    st.warning(f"⏳ Rate limited. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error(f"⚠️ Yahoo Finance rate limit reached for {symbol}. Please try again in a few minutes, or use manual price entry.")
                    return None
            else:
                # Other error
                st.error(f"Error fetching data for {symbol}: {str(e)}")
                return None
    
    return None


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_options_chain(symbol: str, expiration_date: str = None) -> Optional[pd.DataFrame]:
    """Fetch options chain with IV and Open Interest with retry logic"""
    import time
    
    max_retries = 2
    retry_delay = 3
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                return None
            
            # Use provided date or nearest expiration
            exp_date = expiration_date if expiration_date in expirations else expirations[0]
            
            # Get options chain
            opt_chain = ticker.option_chain(exp_date)
            
            # Combine calls and puts
            calls = opt_chain.calls.copy()
            calls['type'] = 'call'
            puts = opt_chain.puts.copy()
            puts['type'] = 'put'
            
            options_df = pd.concat([calls, puts], ignore_index=True)
            
            # Select relevant columns
            columns_to_keep = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 
                              'openInterest', 'impliedVolatility', 'type', 'inTheMoney']
            
            available_columns = [col for col in columns_to_keep if col in options_df.columns]
            options_df = options_df[available_columns]
            
            # Rename for clarity
            options_df = options_df.rename(columns={
                'lastPrice': 'premium',
                'impliedVolatility': 'iv',
                'openInterest': 'open_interest'
            })
            
            return options_df
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if ('rate' in error_msg or '429' in error_msg or 'too many' in error_msg) and attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                time.sleep(wait_time)
                continue
            else:
                if attempt == max_retries - 1:
                    st.warning(f"⚠️ Could not fetch options chain (rate limited or unavailable). Using manual entry.")
                return None
    
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour (expirations don't change often)
def get_available_expirations(symbol: str) -> List[str]:
    """Get available option expiration dates for a symbol with retry logic"""
    import time
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            options = ticker.options
            if options:
                return list(options)
            return []
        except Exception as e:
            error_msg = str(e).lower()
            if ('rate' in error_msg or '429' in error_msg) and attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return []
    return []

def find_nearest_option(options_df: pd.DataFrame, strike: float, option_type: str) -> Optional[Dict]:
    """Find the nearest option contract and return its details including IV and OI"""
    if options_df is None or options_df.empty:
        return None
    
    # Filter by type
    filtered = options_df[options_df['type'] == option_type].copy()
    
    if filtered.empty:
        return None
    
    # Find nearest strike
    filtered['strike_diff'] = abs(filtered['strike'] - strike)
    nearest = filtered.loc[filtered['strike_diff'].idxmin()]
    
    return {
        'strike': nearest['strike'],
        'premium': nearest.get('premium', 0),
        'iv': nearest.get('iv', 0) * 100 if 'iv' in nearest else 0,  # Convert to percentage
        'open_interest': nearest.get('open_interest', 0),
        'bid': nearest.get('bid', 0),
        'ask': nearest.get('ask', 0),
        'volume': nearest.get('volume', 0)
    }

# ==================== ADVANCED ANALYTICS FUNCTIONS ====================

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    if T <= 0 or sigma <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate all Greeks for an option"""
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:  # put
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    theta_part1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        theta = (theta_part1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:  # put
        theta = (theta_part1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega (same for calls and puts)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for 1% change
    
    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:  # put
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def calculate_strategy_greeks(strategy: 'OptionStrategy', risk_free_rate=0.05):
    """Calculate combined Greeks for entire strategy"""
    total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    for leg in strategy.legs:
        T = leg.expiration_days / 365.0
        sigma = 0.30  # Default 30% IV, will be updated with real IV if available
        
        greeks = calculate_greeks(
            S=strategy.stock_price,
            K=leg.strike,
            T=T,
            r=risk_free_rate,
            sigma=sigma,
            option_type=leg.type
        )
        
        # Adjust for buy/sell and quantity
        multiplier = leg.quantity if leg.action == 'buy' else -leg.quantity
        
        for key in total_greeks:
            total_greeks[key] += greeks[key] * multiplier * 100  # Per contract
    
    # Add stock position Greeks if any
    if strategy.underlying_shares != 0:
        total_greeks['delta'] += strategy.underlying_shares
    
    return total_greeks

@st.cache_data(ttl=3600)
def fetch_historical_iv(symbol: str, days=252):
    """Fetch historical IV data with retry logic"""
    import time
    
    max_retries = 2
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            if hist.empty:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return None
            
            # Calculate historical volatility
            hist['returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
            hist_vol = hist['returns'].std() * np.sqrt(252)
            
            return {
                'current_hv': hist_vol,
                'history': hist['returns'].rolling(window=20).std() * np.sqrt(252)
            }
        except Exception as e:
            error_msg = str(e).lower()
            if ('rate' in error_msg or '429' in error_msg) and attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return None
    return None

def calculate_iv_rank_percentile(current_iv, options_chain):
    """Calculate IV Rank and IV Percentile from options chain"""
    if options_chain is None or options_chain.empty or 'iv' not in options_chain.columns:
        return None
    
    iv_values = options_chain['iv'].dropna() * 100  # Convert to percentage
    
    if len(iv_values) == 0:
        return None
    
    iv_min = iv_values.min()
    iv_max = iv_values.max()
    iv_mean = iv_values.mean()
    
    # IV Rank: Where current IV sits in 52-week range (0-100)
    if iv_max > iv_min:
        iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100
    else:
        iv_rank = 50
    
    # IV Percentile: Percentage of days IV was below current (0-100)
    iv_percentile = (iv_values < current_iv).sum() / len(iv_values) * 100
    
    return {
        'iv_rank': iv_rank,
        'iv_percentile': iv_percentile,
        'iv_min': iv_min,
        'iv_max': iv_max,
        'iv_mean': iv_mean,
        'current_iv': current_iv
    }

def calculate_probability_of_profit(strategy: 'OptionStrategy', iv=0.30):
    """Calculate Probability of Profit using Monte Carlo simulation"""
    # Simplified POP calculation based on breakeven points
    metrics = strategy.get_metrics()
    breakevens = metrics['breakeven_points']
    
    if not breakevens:
        # For strategies with no breakeven (always profit or always loss)
        if metrics['max_loss'] > 0:
            return 100.0
        elif metrics['max_profit'] < 0:
            return 0.0
        return 50.0
    
    # Calculate expected move based on IV
    days = 30  # Assume 30 days for simplicity
    expected_move = strategy.stock_price * iv * np.sqrt(days / 365)
    
    # Calculate probability based on standard deviation
    if len(breakevens) == 1:
        # One breakeven point
        z_score = abs(breakevens[0] - strategy.stock_price) / expected_move
        if metrics['max_profit'] > abs(metrics['max_loss']):
            pop = norm.cdf(z_score) * 100
        else:
            pop = (1 - norm.cdf(z_score)) * 100
    else:
        # Two breakeven points (typical for many strategies)
        lower_be = min(breakevens)
        upper_be = max(breakevens)
        
        z_lower = (lower_be - strategy.stock_price) / expected_move
        z_upper = (upper_be - strategy.stock_price) / expected_move
        
        # Probability stock stays between breakevens
        pop = (norm.cdf(z_upper) - norm.cdf(z_lower)) * 100
    
    return max(0, min(100, pop))  # Clamp between 0 and 100

def calculate_expected_move(stock_price, iv, days, confidence=0.68):
    """
    Calculate expected move for a stock
    confidence=0.68 for 1 standard deviation (~68%)
    confidence=0.95 for 2 standard deviations (~95%)
    """
    # Convert annual IV to period IV
    period_iv = iv * np.sqrt(days / 365)
    
    # Expected move
    if confidence == 0.68:
        # 1 standard deviation
        std_devs = 1
    elif confidence == 0.95:
        # 2 standard deviations
        std_devs = 2
    else:
        # Custom confidence level
        std_devs = norm.ppf((1 + confidence) / 2)
    
    move = stock_price * period_iv * std_devs
    
    return {
        'expected_move': move,
        'upper_range': stock_price + move,
        'lower_range': stock_price - move,
        'confidence': confidence * 100
    }

def calculate_max_pain(options_chain, stock_price):
    """Calculate Max Pain - price where option holders experience maximum loss"""
    if options_chain is None or options_chain.empty:
        return None
    
    # Get unique strikes
    strikes = sorted(options_chain['strike'].unique())
    
    if len(strikes) == 0:
        return stock_price
    
    min_pain = float('inf')
    max_pain_strike = stock_price
    
    for strike in strikes:
        total_pain = 0
        
        # Calculate pain for calls
        calls = options_chain[(options_chain['type'] == 'call') & 
                             (options_chain['strike'] == strike)]
        if not calls.empty and 'open_interest' in calls.columns:
            call_oi = calls['open_interest'].sum()
            if strike < stock_price:
                total_pain += call_oi * (stock_price - strike)
        
        # Calculate pain for puts
        puts = options_chain[(options_chain['type'] == 'put') & 
                            (options_chain['strike'] == strike)]
        if not puts.empty and 'open_interest' in puts.columns:
            put_oi = puts['open_interest'].sum()
            if strike > stock_price:
                total_pain += put_oi * (strike - stock_price)
        
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = strike
    
    return max_pain_strike

def calculate_put_call_ratio(options_chain):
    """Calculate Put/Call Ratio by volume and open interest"""
    if options_chain is None or options_chain.empty:
        return None
    
    calls = options_chain[options_chain['type'] == 'call']
    puts = options_chain[options_chain['type'] == 'put']
    
    # Volume-based P/C Ratio
    put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
    call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
    
    pcr_volume = put_volume / call_volume if call_volume > 0 else 0
    
    # Open Interest-based P/C Ratio
    put_oi = puts['open_interest'].sum() if 'open_interest' in puts.columns else 0
    call_oi = calls['open_interest'].sum() if 'open_interest' in calls.columns else 0
    
    pcr_oi = put_oi / call_oi if call_oi > 0 else 0
    
    # Interpretation
    if pcr_oi > 1.0:
        sentiment = "Bearish (More Puts)"
    elif pcr_oi < 0.7:
        sentiment = "Bullish (More Calls)"
    else:
        sentiment = "Neutral"
    
    return {
        'pcr_volume': pcr_volume,
        'pcr_open_interest': pcr_oi,
        'put_volume': put_volume,
        'call_volume': call_volume,
        'put_oi': put_oi,
        'call_oi': call_oi,
        'sentiment': sentiment
    }



def initialize_gemini():
    """Initialize Google Gemini AI"""
    api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Error configuring Gemini API: {str(e)}")
            return False
    return False

def generate_ai_analysis(strategy: OptionStrategy, api_configured: bool, context: Dict = None) -> str:
    """Generate AI-powered analysis of the options strategy"""
    if not api_configured:
        return "⚠️ AI Analysis not available. Please configure GOOGLE_API_KEY in Streamlit secrets."
    
    metrics = strategy.get_metrics()
    greeks = calculate_strategy_greeks(strategy)
    
    # Prepare context string
    context_str = ""
    if context:
        if context.get('stock_data'):
            sd = context['stock_data']
            context_str += f"\nMarket Data:\n- Volume: {sd.get('volume', 'N/A')}\n- Change: {sd.get('change_percent', 0):.2f}%\n- Market Cap: {sd.get('market_cap', 'N/A')}\n"
        
        if context.get('iv_stats'):
            ivs = context['iv_stats']
            context_str += f"\nVolatility Analysis:\n- IV Rank: {ivs.get('iv_rank', 'N/A')}\n- IV Percentile: {ivs.get('iv_percentile', 'N/A')}\n- Current IV: {ivs.get('current_iv', 'N/A')}%\n"
            
        if context.get('pcr'):
            pcr = context['pcr']
            context_str += f"\nOptions Flow:\n- Put/Call Ratio (Vol): {pcr.get('pcr_volume', 'N/A')}\n- Put/Call Ratio (OI): {pcr.get('pcr_open_interest', 'N/A')}\n- Sentiment: {pcr.get('sentiment', 'N/A')}\n"
        if context.get('max_pain'):
            context_str += f"\nMax Pain: ${context['max_pain']:.2f}\n"

    prompt = f"""
    You are ZMtech AI Options Calculator Pro, acting as an institutional-grade options research analyst
    used by hedge funds, prop desks, and professional traders.

    Your task is to generate a decision-first, professionally formatted options strategy report.
    The report must prioritize clarity, probability, and risk management over theoretical payoff.

    DO NOT repeat sections.
    DO NOT bury conclusions.
    Use clean section headers, tables, and concise bullet points.
    Reject poor trades explicitly when justified.

    Use the exact structure and formatting below.

    DATA PROVIDED:
    
    Strategy: {strategy.name}
    Symbol: {strategy.symbol}
    Current Stock Price: ${strategy.stock_price:.2f}
    Date: {datetime.now().strftime('%Y-%m-%d')}
    
    Strategy Legs:
    {chr(10).join([f"- {leg.action.upper()} {leg.quantity} {leg.type.upper()} @ ${leg.strike:.2f}, Premium: ${leg.premium:.2f}" for leg in strategy.legs])}
    
    Strategy Metrics:
    - Maximum Profit: ${metrics['max_profit']:.2f}
    - Maximum Loss: ${metrics['max_loss']:.2f}
    - Net Premium: ${metrics['net_premium']:.2f}
    - Risk/Reward Ratio: {metrics['risk_reward_ratio']:.2f}
    - Breakeven Points: {', '.join([f"${bp:.2f}" for bp in metrics['breakeven_points']]) if metrics['breakeven_points'] else 'None'}
    
    Greeks Data (Total Strategy):
    - Delta: {greeks['delta']:.2f}
    - Gamma: {greeks['gamma']:.4f}
    - Theta: {greeks['theta']:.2f}
    - Vega: {greeks['vega']:.2f}
    - Rho: {greeks['rho']:.2f}
    
    {context_str}

    ==================================================
    ZMtech AI Options Calculator Pro – Institutional Options Strategy Report
    ==================================================

    Ticker: {strategy.symbol}
    Strategy Evaluated: {strategy.name}
    Current Stock Price: ${strategy.stock_price:.2f}
    Report Date: {datetime.now().strftime('%Y-%m-%d')}
    Time Horizon: <SHORT / MEDIUM / LONG>

    --------------------------------------------------
    0. EXECUTIVE SUMMARY (MANDATORY)
    --------------------------------------------------
    Provide a one-screen summary that allows a trader or PM to make a decision in under 60 seconds.

    Include:
    - Strategy evaluated
    - Market bias (Bullish / Bearish / Neutral)
    - Volatility regime (Low / Normal / High / Extreme)
    - Final recommendation (BUY / SELL / HOLD / AVOID)
    - Probability of profit (High / Medium / Low)
    - Dominant risk factor (Theta / IV / Direction / Event)

    Use bullet points.
    Be direct.
    State whether the trade is justified or not.

    --------------------------------------------------
    1. TRADE SETUP SNAPSHOT
    --------------------------------------------------
    Present a quick-glance trade card.

    Include:
    - Underlying price
    - Strategy
    - Strikes
    - Expiration (or state if unavailable)
    - Net premium (total position)
    - Breakeven
    - Required % move
    - IV environment
    - Daily theta impact

    --------------------------------------------------
    2. KEY METRICS SNAPSHOT (TABLE)
    --------------------------------------------------
    Use a clean table with:
    - Maximum Profit
    - Maximum Loss
    - Net Premium
    - Breakeven Point(s)
    - Risk/Reward Ratio

    Add a brief interpretation line under the table.

    --------------------------------------------------
    3. MARKET & VOLATILITY ENVIRONMENT
    --------------------------------------------------
    Analyze:
    - Recent price action
    - Trend direction
    - Options sentiment (Put/Call ratios, OI vs Volume)
    - Implied Volatility (Current IV, IV Rank, IV Percentile)
    - Risk of IV expansion vs IV crush

    Clearly state whether conditions SUPPORT or CONFLICT with the strategy.

    --------------------------------------------------
    4. GREEKS & EXPOSURE PROFILE
    --------------------------------------------------
    Explain Delta ({greeks['delta']:.2f}), Gamma ({greeks['gamma']:.4f}), Theta ({greeks['theta']:.2f}), Vega ({greeks['vega']:.2f}), and Rho ({greeks['rho']:.2f}) in P&L terms.
    Clarify whether values are per-contract or total position.
    Emphasize dominant exposure risks.

    --------------------------------------------------
    5. STRIKE & BREAKEVEN REALISM
    --------------------------------------------------
    Assess whether:
    - Strike selection is realistic
    - Breakeven is achievable
    - Required move aligns with trend and timeframe

    Quantify the required move and compare it to recent price behavior.

    --------------------------------------------------
    6. RISK ANALYSIS
    --------------------------------------------------
    Identify:
    - Maximum capital at risk
    - Time decay risk
    - Volatility risk
    - Directional risk
    - Event risk (earnings, macro, news)

    Provide practical risk-management guidance:
    - Position sizing
    - Stop-loss logic
    - Time-based exits

    --------------------------------------------------
    7. PROBABILITY & EXPECTED VALUE
    --------------------------------------------------
    Assess:
    - Probability of finishing ITM
    - Probability of reaching breakeven
    - Likelihood of max profit

    Explicitly answer:
    “Does probability justify this trade? YES or NO”

    --------------------------------------------------
    8. ALTERNATIVE OPTIONS STRATEGIES
    --------------------------------------------------
    For each strategy, include:
    - Strategy name
    - Setup (strikes, expiration)
    - Objective
    - Risk
    - Best timing

    Include:
    - Long Call
    - Long Put
    - Bull Call Spread
    - Cash-Secured Put
    - Protective Put
    - Volatility strategy (Iron Condor or Straddle)

    --------------------------------------------------
    9. STRATEGY RECOMMENDATION
    --------------------------------------------------
    Select ONE preferred strategy.

    Justify using:
    - Trend alignment
    - Volatility considerations
    - Capital efficiency
    - Risk control
    - Time horizon

    Add tactical notes:
    - IV behavior
    - Earnings or event risk
    - Position sizing guidance

    --------------------------------------------------
    10. TRADE SIGNAL
    --------------------------------------------------
    Clearly state:
    BUY / SELL / HOLD / AVOID

    Summarize the decision in 3–5 bullet points.

    --------------------------------------------------
    11. KEY LEVELS & EXIT PLAN
    --------------------------------------------------
    Include:
    - Key price levels
    - Profit-taking guidance
    - Stop-loss triggers
    - Time-based exits
    - IV-driven exits

    --------------------------------------------------
    12. FINAL CONCLUSION
    --------------------------------------------------
    Deliver a firm, professional verdict.
    Do not hedge language.
    State clearly whether the trade is worth taking or not.

    --------------------------------------------------
    13. DISCLAIMER
    --------------------------------------------------
    Include a standard informational disclaimer indicating this is not financial advice.
    """
    
    # Try different model names (Google Gemini 2.5 is the latest)
    model_names = ['gemini-2.5-flash', 'gemini-flash-latest', 'gemini-2.0-flash', 'gemini-pro-latest']
    
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                continue  # Try next model
            else:
                # Different error, report it
                return f"⚠️ Error generating AI analysis: {error_msg}"
    
    # If all models failed, try to list available models
    try:
        available_models = list(genai.list_models())
        model_list = "\n".join([f"- {m.name}" for m in available_models if 'generateContent' in m.supported_generation_methods])
        return f"""⚠️ Could not find a working Gemini model. 

**Available models that support generateContent:**
{model_list}

**Troubleshooting:**
1. Verify your API key is correct in .streamlit/secrets.toml
2. Make sure you have access to Gemini API at https://aistudio.google.com/
3. Check if you need to enable the API in your Google Cloud Console

**Tried models:** {', '.join(model_names)}"""
    except Exception as e:
        return f"⚠️ Error accessing Gemini API: {str(e)}\n\nPlease verify your API key and internet connection."
def create_payoff_chart(strategy: OptionStrategy):
    """Create interactive payoff diagram"""
    price_range = np.linspace(strategy.stock_price * 0.5, strategy.stock_price * 1.5, 1000)
    payoff = strategy.calculate_payoff(price_range)
    
    fig = go.Figure()
    
    # Add payoff line
    fig.add_trace(go.Scatter(
        x=price_range,
        y=payoff,
        mode='lines',
        name='Payoff',
        line=dict(color='#667eea', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    # Add zero line
    fig.add_trace(go.Scatter(
        x=price_range,
        y=np.zeros_like(price_range),
        mode='lines',
        name='Break-even',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash')
    ))
    
    # Add current price line
    fig.add_vline(
        x=strategy.stock_price,
        line=dict(color='#fbbf24', width=2, dash='dot'),
        annotation_text=f"Current: ${strategy.stock_price:.2f}",
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        title=f"{strategy.name} - Profit/Loss at Expiration",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit/Loss ($)",
        template="plotly_dark",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12),
        showlegend=True,
        height=500
    )
    
    return fig

def get_strategy_template(strategy_name: str, stock_price: float) -> List[OptionLeg]:
    """Get predefined option legs for common strategies"""
    templates = {
        "Long Call": [
            OptionLeg("call", "buy", stock_price * 1.05, stock_price * 0.05, 1, 30)
        ],
        "Long Put": [
            OptionLeg("put", "buy", stock_price * 0.95, stock_price * 0.05, 1, 30)
        ],
        "Covered Call": [
            OptionLeg("call", "sell", stock_price * 1.10, stock_price * 0.04, 1, 30)
        ],
        "Cash Secured Put": [
            OptionLeg("put", "sell", stock_price * 0.95, stock_price * 0.04, 1, 30)
        ],
        "Naked Call": [
            OptionLeg("call", "sell", stock_price * 1.10, stock_price * 0.05, 1, 30)
        ],
        "Naked Put": [
            OptionLeg("put", "sell", stock_price * 0.90, stock_price * 0.05, 1, 30)
        ],
        "Bull Call Spread": [
            OptionLeg("call", "buy", stock_price, stock_price * 0.05, 1, 30),
            OptionLeg("call", "sell", stock_price * 1.10, stock_price * 0.02, 1, 30)
        ],
        "Bear Put Spread": [
            OptionLeg("put", "buy", stock_price, stock_price * 0.05, 1, 30),
            OptionLeg("put", "sell", stock_price * 0.90, stock_price * 0.02, 1, 30)
        ],
        "Iron Condor": [
            OptionLeg("put", "buy", stock_price * 0.90, stock_price * 0.02, 1, 30),
            OptionLeg("put", "sell", stock_price * 0.95, stock_price * 0.04, 1, 30),
            OptionLeg("call", "sell", stock_price * 1.05, stock_price * 0.04, 1, 30),
            OptionLeg("call", "buy", stock_price * 1.10, stock_price * 0.02, 1, 30)
        ],
        "Butterfly": [
            OptionLeg("call", "buy", stock_price * 0.95, stock_price * 0.06, 1, 30),
            OptionLeg("call", "sell", stock_price, stock_price * 0.04, 2, 30),
            OptionLeg("call", "buy", stock_price * 1.05, stock_price * 0.02, 1, 30)
        ],
        "Straddle": [
            OptionLeg("call", "buy", stock_price, stock_price * 0.05, 1, 30),
            OptionLeg("put", "buy", stock_price, stock_price * 0.05, 1, 30)
        ],
        "Strangle": [
            OptionLeg("call", "buy", stock_price * 1.05, stock_price * 0.04, 1, 30),
            OptionLeg("put", "buy", stock_price * 0.95, stock_price * 0.04, 1, 30)
        ],
        "Collar": [
            OptionLeg("call", "sell", stock_price * 1.10, stock_price * 0.03, 1, 30),
            OptionLeg("put", "buy", stock_price * 0.90, stock_price * 0.03, 1, 30)
        ],
        # Additional Spread Strategies
        "Credit Spread": [
            # Bull Put Credit Spread (collect credit, bullish)
            OptionLeg("put", "sell", stock_price * 0.98, stock_price * 0.03, 1, 30),
            OptionLeg("put", "buy", stock_price * 0.93, stock_price * 0.01, 1, 30)
        ],
        "Call Spread": [
            # Debit Call Spread (bullish)
            OptionLeg("call", "buy", stock_price * 1.02, stock_price * 0.04, 1, 30),
            OptionLeg("call", "sell", stock_price * 1.08, stock_price * 0.02, 1, 30)
        ],
        "Put Spread": [
            # Debit Put Spread (bearish)
            OptionLeg("put", "buy", stock_price * 0.98, stock_price * 0.04, 1, 30),
            OptionLeg("put", "sell", stock_price * 0.92, stock_price * 0.02, 1, 30)
        ],
        "Calendar Spread": [
            # Buy longer-dated (more time), Sell shorter-dated (less time) - Same strike
            OptionLeg("call", "buy", stock_price, stock_price * 0.06, 1, 60),   # Long-term (Buy)
            OptionLeg("call", "sell", stock_price, stock_price * 0.04, 1, 30)   # Short-term (Sell)
        ],
        "Ratio Back Spread": [
            # Sell 1 ATM, Buy 2 OTM (profit from big move)
            OptionLeg("call", "sell", stock_price, stock_price * 0.05, 1, 30),
            OptionLeg("call", "buy", stock_price * 1.10, stock_price * 0.02, 2, 30)
        ],
        # Advanced strategies already included above
        "Diagonal Spread": [
            # Different strikes AND different expirations
            OptionLeg("call", "sell", stock_price * 1.05, stock_price * 0.04, 1, 30),
            OptionLeg("call", "buy", stock_price * 1.10, stock_price * 0.03, 1, 60)
        ],
        "Double Diagonal": [
            # Iron Condor with different expirations
            OptionLeg("put", "sell", stock_price * 0.95, stock_price * 0.03, 1, 30),
            OptionLeg("put", "buy", stock_price * 0.90, stock_price * 0.02, 1, 60),
            OptionLeg("call", "sell", stock_price * 1.05, stock_price * 0.03, 1, 30),
            OptionLeg("call", "buy", stock_price * 1.10, stock_price * 0.02, 1, 60)
        ]
    }
    
    return templates.get(strategy_name, [])

def main():
    load_css()
    
    # Initialize AI model
    ai_model = initialize_gemini()
    
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: white; margin-bottom: 10px;'>
            📈 ZMtech AI Options Calculator Pro
        </h1>
        <p style='text-align: center; color: #a0aec0; font-size: 18px; margin-bottom: 40px;'>
            Advanced Options Trading Analysis powered by AI
        </p>
    """, unsafe_allow_html=True)
    
    # Sidebar for strategy selection
    with st.sidebar:
        st.markdown("## 🎯 Select Strategy")
        
        strategy_category = st.selectbox(
            "Category",
            ["Basic", "Spreads", "Advanced", "Custom"]
        )
        
        # Strategy selection based on category
        if strategy_category == "Basic":
            strategies = [
                "Long Call", "Long Put", "Covered Call",
                "Cash Secured Put", "Naked Call", "Naked Put"
            ]
        elif strategy_category == "Spreads":
            strategies = [
                "Bull Call Spread", "Bear Put Spread", "Credit Spread",
                "Call Spread", "Put Spread", "Calendar Spread", "Ratio Back Spread"
            ]
        elif strategy_category == "Advanced":
            strategies = [
                "Iron Condor", "Butterfly", "Collar", "Straddle",
                "Strangle", "Diagonal Spread", "Double Diagonal"
            ]
        else:  # Custom
            strategies = ["Custom 2 Legs", "Custom 3 Legs", "Custom 4 Legs", 
                         "Custom 5 Legs", "Custom 6 Legs", "Custom 8 Legs"]
        
        selected_strategy = st.selectbox("Strategy", strategies)
        
        st.markdown("---")
        st.markdown("## 📊 Market Data")
        
        # Stock symbol input
        use_real_data = st.checkbox("📈 Use Real-Time Market Data", value=True)
        
        stock_symbol = ""
        stock_data = None
        options_chain = None
        selected_expiration = None
        iv_stats = None
        pcr = None
        max_pain = None
        
        if use_real_data:
            stock_symbol = st.text_input(
                "Stock Symbol",
                value="AAPL",
                help="Enter stock ticker symbol (e.g., AAPL, TSLA, SPY)"
            ).upper()
            
            if stock_symbol:
                with st.spinner(f"Fetching data for {stock_symbol}..."):
                    stock_data = fetch_stock_data(stock_symbol)
                    
                if stock_data:
                    # Display stock info
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            label=stock_data['company_name'],
                            value=f"${stock_data['price']:.2f}",
                            delta=f"{stock_data['change_percent']:.2f}%"
                        )
                    with col_b:
                        st.metric(
                            label="Volume",
                            value=f"{stock_data['volume']:,.0f}"
                        )
                    
                    # Get available expirations
                    expirations = get_available_expirations(stock_symbol)
                    if expirations:
                        selected_expiration = st.selectbox(
                            "Option Expiration Date",
                            options=expirations,
                            help="Select the expiration date for options data"
                        )
                        
                        # Fetch options chain
                        options_chain = fetch_options_chain(stock_symbol, selected_expiration)
                        
                        if options_chain is not None and not options_chain.empty:
                            st.success(f"✅ Loaded {len(options_chain)} option contracts")
                    
                    stock_price = stock_data['price']
                else:
                    st.error(f"Could not fetch data for {stock_symbol}. Using manual input.")
                    stock_price = st.number_input(
                        "Current Stock Price ($)",
                        min_value=1.0,
                        value=100.0,
                        step=1.0
                    )
            else:
                stock_price = 100.0
        else:
            stock_price = st.number_input(
                "Current Stock Price ($)",
                min_value=1.0,
                value=100.0,
                step=1.0
            )
        
        # For covered strategies
        has_stock = st.checkbox("Include Stock Position")
        stock_shares = 0
        if has_stock:
            stock_shares = st.number_input(
                "Number of Shares",
                min_value=1,
                value=100,
                step=1
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔧 Configure Option Legs")
        
        # Get or create strategy legs
        if "Custom" not in selected_strategy:
            legs = get_strategy_template(selected_strategy, stock_price)
        else:
            num_legs = int(selected_strategy.split()[1])
            legs = []
            
            for i in range(num_legs):
                st.markdown(f"**Leg {i+1}**")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    leg_type = st.selectbox(f"Type###{i}", ["call", "put"], key=f"type_{i}")
                    leg_action = st.selectbox(f"Action###{i}", ["buy", "sell"], key=f"action_{i}")
                
                with col_b:
                    strike = st.number_input(
                        f"Strike###{i}",
                        min_value=1.0,
                        value=stock_price * (1.0 + 0.05 * i),
                        step=1.0,
                        key=f"strike_{i}"
                    )
                    premium = st.number_input(
                        f"Premium###{i}",
                        min_value=0.01,
                        value=stock_price * 0.04,
                        step=0.01,
                        key=f"premium_{i}"
                    )
                
                with col_c:
                    quantity = st.number_input(
                        f"Quantity###{i}",
                        min_value=1,
                        value=1,
                        step=1,
                        key=f"quantity_{i}"
                    )
                    expiration = st.number_input(
                        f"Days to Exp###{i}",
                        min_value=1,
                        value=30,
                        step=1,
                        key=f"expiration_{i}"
                    )
                
                legs.append(OptionLeg(leg_type, leg_action, strike, premium, quantity, expiration))
                st.markdown("---")
        
        # Allow editing of template strategies
        if "Custom" not in selected_strategy and st.checkbox("Customize This Strategy"):
            st.markdown("#### Adjust Parameters")
            edited_legs = []
            
            for i, leg in enumerate(legs):
                st.markdown(f"**Leg {i+1}** - {leg.action.upper()} {leg.type.upper()}")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    strike = st.number_input(
                        f"Strike###{i}_edit",
                        min_value=1.0,
                        value=float(leg.strike),
                        step=1.0,
                        key=f"strike_{i}_edit"
                    )
                
                with col_b:
                    premium = st.number_input(
                        f"Premium###{i}_edit",
                        min_value=0.01,
                        value=float(leg.premium),
                        step=0.01,
                        key=f"premium_{i}_edit"
                    )
                
                with col_c:
                    quantity = st.number_input(
                        f"Quantity###{i}_edit",
                        min_value=1,
                        value=leg.quantity,
                        step=1,
                        key=f"quantity_{i}_edit"
                    )
                    expiration = st.number_input(
                        f"Days to Exp###{i}_edit",
                        min_value=1,
                        value=leg.expiration_days,
                        step=1,
                        key=f"expiration_{i}_edit"
                    )
                
                edited_legs.append(OptionLeg(
                    leg.type, leg.action, strike, premium, quantity, expiration
                ))
            
            legs = edited_legs
        
        # Create strategy object
        strategy = OptionStrategy(
            name=selected_strategy,
            legs=legs,
            stock_price=stock_price,
            underlying_shares=stock_shares,
            symbol=stock_symbol if stock_symbol else "MANUAL"
        )
        
        # Calculate and display metrics
        metrics = strategy.get_metrics()
        
        st.markdown("### 📊 Payoff Diagram")
        chart = create_payoff_chart(strategy)
        st.plotly_chart(chart, use_container_width=True)
        
        # Display options data if available
        if use_real_data and options_chain is not None and not options_chain.empty:
            st.markdown("### 📋 Options Chain Data")
            
            # Show data for each leg
            for i, leg in enumerate(legs):
                option_data = find_nearest_option(options_chain, leg.strike, leg.type)
                
                if option_data:
                    with st.expander(f"**Leg {i+1}** - {leg.action.upper()} {leg.type.upper()} @ ${leg.strike:.2f}"):
                        col_x, col_y, col_z = st.columns(3)
                        
                        with col_x:
                            st.metric("Actual Strike", f"${option_data['strike']:.2f}")
                            st.metric("Premium", f"${option_data['premium']:.2f}")
                        
                        with col_y:
                            st.metric("IV", f"{option_data['iv']:.1f}%")
                            st.metric("Bid/Ask", f"${option_data['bid']:.2f}/{option_data['ask']:.2f}")
                        
                        with col_z:
                            st.metric("Open Interest", f"{option_data['open_interest']:,.0f}")
                            st.metric("Volume", f"{option_data['volume']:,.0f}")
                        
                        # Update button to use live data
                        if st.button(f"Use Live Data for Leg {i+1}", key=f"use_live_{i}"):
                            st.info(f"💡 Tip: Manually adjust the premium to ${option_data['premium']:.2f} and strike to ${option_data['strike']:.2f} in the configuration above")
    
    with col2:
        st.markdown("### 📈 Key Metrics")
        
        # Display metrics in cards
        st.markdown(f"""
            <div class="metric-card">
                <h4 style='margin:0; color: white;'>Max Profit</h4>
                <p style='font-size: 24px; font-weight: 700; margin: 10px 0;'>
                    ${metrics['max_profit']:.2f}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h4 style='margin:0; color: white;'>Max Loss</h4>
                <p style='font-size: 24px; font-weight: 700; margin: 10px 0;'>
                    ${metrics['max_loss']:.2f}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h4 style='margin:0; color: white;'>Net Premium</h4>
                <p style='font-size: 24px; font-weight: 700; margin: 10px 0;'>
                    ${metrics['net_premium']:.2f}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                <h4 style='margin:0; color: white;'>Risk/Reward</h4>
                <p style='font-size: 24px; font-weight: 700; margin: 10px 0;'>
                    {metrics['risk_reward_ratio']:.2f}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if metrics['breakeven_points']:
            st.markdown("**Breakeven Points:**")
            for bp in metrics['breakeven_points']:
                st.markdown(f"- ${bp:.2f}")
        
        # Advanced Analytics Section
        st.markdown("---")
        st.markdown("### 🎯 Advanced Analytics")
        
        # Calculate Greeks
        greeks = calculate_strategy_greeks(strategy)
        
        with st.expander("📊 **Greeks Analysis**", expanded=True):
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.metric("Delta (Δ)", f"{greeks['delta']:.2f}", 
                         help="Position delta - directional exposure")
                st.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}",
                         help="Rate of change of delta")
                st.metric("Theta (Θ)", f"${greeks['theta']:.2f}/day",
                         help="Daily time decay")
            
            with col_g2:
                st.metric("Vega (V)", f"${greeks['vega']:.2f}",
                         help="Sensitivity to 1% IV change")
                st.metric("Rho (ρ)", f"${greeks['rho']:.2f}",
                         help="Sensitivity to 1% interest rate change")
        
        # Probability of Profit
        pop = calculate_probability_of_profit(strategy, iv=0.30)
        st.metric("📈 Probability of Profit", f"{pop:.1f}%",
                 help="Estimated probability this trade will be profitable")
        
        # Expected Move (if using real data)
        if use_real_data and stock_data:
            expected_move_1sd = calculate_expected_move(stock_price, 0.30, 30, 0.68)
            expected_move_2sd = calculate_expected_move(stock_price, 0.30, 30, 0.95)
            
            with st.expander("📏 **Expected Move (30 days)**"):
                st.markdown(f"""
                **1 Standard Deviation (68% confidence):**
                - Range: ${expected_move_1sd['lower_range']:.2f} - ${expected_move_1sd['upper_range']:.2f}
                - Move: ±${expected_move_1sd['expected_move']:.2f}
                
                **2 Standard Deviations (95% confidence):**
                - Range: ${expected_move_2sd['lower_range']:.2f} - ${expected_move_2sd['upper_range']:.2f}
                - Move: ±${expected_move_2sd['expected_move']:.2f}
                """)
        
        # IV Rank and Percentile (if using real data with options chain)
        if use_real_data and options_chain is not None:
            # Get average IV from options chain
            avg_iv = options_chain['iv'].mean() * 100 if 'iv' in options_chain.columns else None
            
            if avg_iv:
                iv_stats = calculate_iv_rank_percentile(avg_iv, options_chain)
                
                if iv_stats:
                    with st.expander("📈 **IV Analysis**"):
                        col_iv1, col_iv2 = st.columns(2)
                        
                        with col_iv1:
                            st.metric("IV Rank", f"{iv_stats['iv_rank']:.1f}",
                                     help="Where current IV sits in 52-week range (0-100)")
                            st.metric("IV Percentile", f"{iv_stats['iv_percentile']:.1f}%",
                                     help="% of time IV was below current level")
                        
                        with col_iv2:
                            st.metric("Current IV", f"{iv_stats['current_iv']:.1f}%")
                            st.metric("IV Range", f"{iv_stats['iv_min']:.1f}% - {iv_stats['iv_max']:.1f}%")
                        
                        # IV interpretation
                        if iv_stats['iv_rank'] > 70:
                            st.info("🔥 **High IV** - Good for selling premium")
                        elif iv_stats['iv_rank'] < 30:
                            st.info("❄️ **Low IV** - Good for buying options")
                        else:
                            st.info("😐 **Neutral IV** - Average conditions")
            
            # Max Pain
            max_pain = calculate_max_pain(options_chain, stock_price)
            if max_pain:
                st.metric("🎯 Max Pain Strike", f"${max_pain:.2f}",
                         help="Price where option holders experience maximum loss")
                
                distance_to_max_pain = ((stock_price - max_pain) / stock_price) * 100
                if abs(distance_to_max_pain) < 2:
                    st.success(f"Stock is near Max Pain (within 2%)")
                elif distance_to_max_pain > 0:
                    st.info(f"Stock is {abs(distance_to_max_pain):.1f}% above Max Pain")
                else:
                    st.info(f"Stock is {abs(distance_to_max_pain):.1f}% below Max Pain")
            
            # Put/Call Ratio
            pcr = calculate_put_call_ratio(options_chain)
            if pcr:
                with st.expander("⚖️ **Put/Call Ratio**"):
                    col_pcr1, col_pcr2 = st.columns(2)
                    
                    with col_pcr1:
                        st.metric("P/C by Volume", f"{pcr['pcr_volume']:.2f}")
                        st.metric("Put Volume", f"{pcr['put_volume']:,.0f}")
                        st.metric("Call Volume", f"{pcr['call_volume']:,.0f}")
                    
                    with col_pcr2:
                        st.metric("P/C by OI", f"{pcr['pcr_open_interest']:.2f}")
                        st.metric("Put OI", f"{pcr['put_oi']:,.0f}")
                        st.metric("Call OI", f"{pcr['call_oi']:,.0f}")
                    
                    # Sentiment
                    if pcr['sentiment'] == "Bearish (More Puts)":
                        st.error(f"📉 {pcr['sentiment']}")
                    elif pcr['sentiment'] == "Bullish (More Calls)":
                        st.success(f"📈 {pcr['sentiment']}")
                    else:
                        st.info(f"😐 {pcr['sentiment']}")
    
    # AI Analysis Section
    st.markdown("---")
    st.markdown("## 🤖 AI-Powered Analysis")

    
    # Initialize session state for analysis if not present
    if 'analysis_content' not in st.session_state:
        st.session_state.analysis_content = None
    if 'analysis_strategy_name' not in st.session_state:
        st.session_state.analysis_strategy_name = None
    if 'analysis_metrics' not in st.session_state:
        st.session_state.analysis_metrics = None

    if st.button("🚀 Generate Comprehensive Report", use_container_width=True):
        with st.spinner("Analyzing strategy with Google Gemini AI..."):
            # Create context for AI
            context = {
                'stock_data': stock_data,
                'iv_stats': iv_stats,
                'pcr': pcr,
                'max_pain': max_pain
            }
            
            analysis = generate_ai_analysis(strategy, ai_model, context)
            st.session_state.analysis_content = analysis
            st.session_state.analysis_strategy_name = strategy.name
            st.session_state.analysis_metrics = strategy.get_metrics()
            
    # Display analysis if available
    if st.session_state.analysis_content:
        analysis = st.session_state.analysis_content
        # Use stored metadata for consistency
        stored_strategy_name = st.session_state.analysis_strategy_name
        stored_metrics = st.session_state.analysis_metrics
        
        st.markdown(f"""
            <div class="ai-report">
                {analysis.replace(chr(10), '<br>')}
            </div>
        """, unsafe_allow_html=True)
        
        # Extract signal if present
        if "BUY" in analysis.upper() and "SELL" not in analysis.upper()[:analysis.upper().find("BUY")]:
            st.markdown("<p class='buy-signal'>🟢 Signal: BUY</p>", unsafe_allow_html=True)
        elif "SELL" in analysis.upper():
            st.markdown("<p class='sell-signal'>🔴 Signal: SELL</p>", unsafe_allow_html=True)
        elif "HOLD" in analysis.upper():
            st.markdown("<p class='hold-signal'>🟡 Signal: HOLD</p>", unsafe_allow_html=True)
            
        # Download Buttons
        st.markdown("### 📥 Download Report")
        d_col1, d_col2, d_col3 = st.columns(3)
        
        # Prepare data for reports using STORED values
        report_title = f"{stored_strategy_name} Analysis"
        
        # Text Download
        with d_col1:
            st.download_button(
                label="📄 Download Text",
                data=analysis,
                file_name=f"{stored_strategy_name.replace(' ', '_')}_analysis.txt",
                mime="text/plain"
            )
            
        # Word Download
        with d_col2:
            try:
                docx_file = create_word_report(report_title, stored_strategy_name, analysis, stored_metrics)
                st.download_button(
                    label="📝 Download Word",
                    data=docx_file,
                    file_name=f"{stored_strategy_name.replace(' ', '_')}_analysis.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"Word generation failed: {e}")

        # PDF Download
        with d_col3:
            try:
                pdf_file = create_pdf_report(report_title, stored_strategy_name, analysis, stored_metrics)
                st.download_button(
                    label="📕 Download PDF",
                    data=pdf_file,
                    file_name=f"{stored_strategy_name.replace(' ', '_')}_analysis.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; color: #a0aec0; font-size: 14px;'>
            ⚠️ This tool is for educational purposes only. Options trading involves significant risk.
            Always consult with a financial advisor before making investment decisions.
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
