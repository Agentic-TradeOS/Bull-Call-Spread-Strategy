"""
Bull Call Spread Strategy
Defined-risk options strategy using two call options at different strikes.

Pricing: Cox-Ross-Rubinstein (CRR) binomial tree — American-style exercise.
Structure: Buy lower-strike call, sell higher-strike call (same expiry)
Max Profit: (high_strike - low_strike - net_debit) * 100
Max Loss:   net_debit * 100

Author: Agentic Trading
Version: 2.0.0
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ─── CRR Binomial Tree ────────────────────────────────────────────────────────

def crr_price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = 'call', steps: int = 100,
              style: str = 'american') -> float:
    """
    Cox-Ross-Rubinstein binomial tree pricing for American or European options.

    Parameters
    ----------
    S     : float  Current spot price
    K     : float  Strike price
    T     : float  Time to expiry in years
    r     : float  Annual risk-free rate (e.g. 0.05)
    sigma : float  Implied/historical volatility (e.g. 0.25)
    option_type : 'call' or 'put'
    steps : int    Number of time steps (default 100)
    style : 'american' or 'european'
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)

    dt   = T / steps
    u    = math.exp(sigma * math.sqrt(dt))   # up factor
    d    = 1.0 / u                            # down factor (recombining tree)
    disc = math.exp(-r * dt)                  # per-step discount
    p    = (math.exp(r * dt) - d) / (u - d)  # risk-neutral up probability

    # Terminal payoffs
    values = [
        max(S * (u ** j) * (d ** (steps - j)) - K, 0.0) if option_type == 'call'
        else max(K - S * (u ** j) * (d ** (steps - j)), 0.0)
        for j in range(steps + 1)
    ]

    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = disc * (p * values[j + 1] + (1 - p) * values[j])
            if style == 'american':
                spot_ij = S * (u ** j) * (d ** (i - j))
                intrinsic = max(spot_ij - K, 0.0) if option_type == 'call' else max(K - spot_ij, 0.0)
                values[j] = max(intrinsic, continuation)
            else:
                values[j] = continuation

    return round(values[0], 6)


def crr_greeks(S: float, K: float, T: float, r: float, sigma: float,
               option_type: str = 'call', steps: int = 100) -> Dict:
    """Greeks via central finite differences on the CRR price."""
    dS    = S * 0.01
    dsig  = 0.01
    dr    = 0.005
    dt    = 1 / 365

    base  = crr_price(S, K, T, r, sigma, option_type, steps)
    pu    = crr_price(S + dS, K, T, r, sigma, option_type, steps)
    pd    = crr_price(S - dS, K, T, r, sigma, option_type, steps)
    pvu   = crr_price(S, K, T, r, sigma + dsig, option_type, steps)
    pvd   = crr_price(S, K, T, r, max(sigma - dsig, 0.01), option_type, steps)
    pru   = crr_price(S, K, T, r + dr, sigma, option_type, steps)
    prd   = crr_price(S, K, T, max(r - dr, 0.001), sigma, option_type, steps)
    pt    = crr_price(S, K, max(T - dt, 1e-6), r, sigma, option_type, steps) if T > dt else base

    return {
        'delta': (pu - pd) / (2 * dS),
        'gamma': (pu - 2 * base + pd) / (dS ** 2),
        'theta': (pt - base) / dt / 365,
        'vega':  (pvu - pvd) / (2 * dsig) / 100,
        'rho':   (pru - prd) / (2 * dr) / 100,
    }


# ─── Strategy ────────────────────────────────────────────────────────────────

@dataclass
class SpreadResult:
    long_call_price: float
    short_call_price: float
    net_debit: float
    max_profit: float
    max_loss: float
    breakeven: float
    risk_reward_ratio: float
    profit_at_expiry: float


class BullCallSpreadStrategy:
    """
    Bull Call Spread Strategy (CRR pricing)

    Buys an ATM/slightly-OTM call and sells a further-OTM call to cap cost.
    Pricing uses the Cox-Ross-Rubinstein binomial tree (American exercise).

    Parameters
    ----------
    low_strike_offset  : % offset for the long call strike  (0.0  = ATM)
    high_strike_offset : % offset for the short call strike (0.05 = 5% OTM)
    days_to_expiry     : days until expiry  (default 30)
    risk_free_rate     : annual risk-free rate (default 0.05)
    implied_volatility : IV override; falls back to historical_vol if None
    contracts          : number of contracts (default 1)
    crr_steps          : binomial tree steps (default 100)
    """

    def __init__(
        self,
        low_strike_offset: float = 0.0,
        high_strike_offset: float = 0.05,
        days_to_expiry: int = 30,
        risk_free_rate: float = 0.05,
        implied_volatility: Optional[float] = None,
        contracts: int = 1,
        crr_steps: int = 100,
    ):
        self.low_strike_offset  = low_strike_offset
        self.high_strike_offset = high_strike_offset
        self.days_to_expiry     = days_to_expiry
        self.risk_free_rate     = risk_free_rate
        self.implied_volatility = implied_volatility
        self.contracts          = contracts
        self.crr_steps          = crr_steps

    def evaluate(self, spot_price: float, historical_vol: float = 0.25,
                 target_price: Optional[float] = None) -> SpreadResult:
        sigma       = self.implied_volatility or historical_vol
        T           = self.days_to_expiry / 365.0
        low_strike  = round(spot_price * (1 + self.low_strike_offset), 2)
        high_strike = round(spot_price * (1 + self.high_strike_offset), 2)

        long_call  = crr_price(spot_price, low_strike,  T, self.risk_free_rate, sigma, 'call', self.crr_steps)
        short_call = crr_price(spot_price, high_strike, T, self.risk_free_rate, sigma, 'call', self.crr_steps)

        net_debit    = (long_call - short_call) * self.contracts * 100
        spread_width = (high_strike - low_strike) * self.contracts * 100
        max_profit   = spread_width - net_debit
        max_loss     = net_debit
        breakeven    = low_strike + (net_debit / (self.contracts * 100))
        risk_reward  = max_profit / max_loss if max_loss > 0 else 0

        tp = target_price or high_strike
        long_pnl  = max(tp - low_strike,  0) - long_call
        short_pnl = short_call - max(tp - high_strike, 0)
        profit_at_target = (long_pnl + short_pnl) * self.contracts * 100

        return SpreadResult(
            long_call_price=long_call, short_call_price=short_call,
            net_debit=net_debit, max_profit=max_profit, max_loss=max_loss,
            breakeven=breakeven, risk_reward_ratio=risk_reward,
            profit_at_expiry=profit_at_target,
        )

    def greeks(self, spot_price: float, historical_vol: float = 0.25) -> Dict:
        sigma       = self.implied_volatility or historical_vol
        T           = self.days_to_expiry / 365.0
        low_strike  = spot_price * (1 + self.low_strike_offset)
        high_strike = spot_price * (1 + self.high_strike_offset)
        lg = crr_greeks(spot_price, low_strike,  T, self.risk_free_rate, sigma, 'call', self.crr_steps)
        sg = crr_greeks(spot_price, high_strike, T, self.risk_free_rate, sigma, 'call', self.crr_steps)
        return {k: lg[k] - sg[k] for k in lg}

    def pnl_at_expiry(self, spot_price: float, price_range: Tuple[float, float], steps: int = 50) -> Dict:
        low_strike  = spot_price * (1 + self.low_strike_offset)
        high_strike = spot_price * (1 + self.high_strike_offset)
        T           = self.days_to_expiry / 365.0
        long_cost   = crr_price(spot_price, low_strike,  T, self.risk_free_rate,
                                self.implied_volatility or 0.25, 'call', self.crr_steps)
        short_cred  = crr_price(spot_price, high_strike, T, self.risk_free_rate,
                                self.implied_volatility or 0.25, 'call', self.crr_steps)
        net_debit = long_cost - short_cred
        lo, hi    = price_range
        prices    = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
        pnls      = [(max(p - low_strike, 0) - max(p - high_strike, 0) - net_debit)
                     * self.contracts * 100 for p in prices]
        return {'prices': prices, 'pnl': pnls}


# ─── Real options chain helpers ─────────────────────────────────────────────

@dataclass
class ChainOption:
    """Single strike row from a live options chain."""
    strike: float
    expiry: str           # YYYY-MM-DD
    bid: float
    ask: float
    iv: float             # implied volatility, e.g. 0.30 = 30%
    delta: float
    volume: int
    open_interest: int
    
    @property
    def mid(self) -> float:
        """Mid-market price (average of bid and ask)."""
        return (self.bid + self.ask) / 2.0


def filter_by_liquidity(chain: list[ChainOption], min_volume: int = 10, 
                        min_open_interest: int = 100) -> list[ChainOption]:
    """Remove illiquid strikes from a chain."""
    return [o for o in chain if o.volume >= min_volume and o.open_interest >= min_open_interest]


def bid_ask_spread_pct(option: ChainOption) -> float:
    """Calculate bid-ask spread as % of mid price."""
    if option.mid == 0:
        return 0.0
    return ((option.ask - option.bid) / option.mid) * 100


def select_strike(target_price: float, available_strikes: list[float]) -> float:
    """Snap target price to nearest real strike in chain."""
    if not available_strikes:
        raise ValueError('available_strikes cannot be empty')
    return min(available_strikes, key=lambda s: abs(s - target_price))


def select_expiry(target_dte: int, available_expiries: list[str], 
                  today: Optional[str] = None) -> str:
    """Find expiration date closest to target DTE (days to expiry)."""
    if not available_expiries:
        raise ValueError('available_expiries cannot be empty')
    
    from datetime import datetime, timedelta
    today_date = datetime.strptime(today, '%Y-%m-%d') if today else datetime.now()
    
    def days_to_expiry(exp_str: str) -> int:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
        return (exp_date - today_date).days
    
    return min(available_expiries, 
               key=lambda exp: abs(days_to_expiry(exp) - target_dte))


@dataclass
class ChainSpreadAnalysis:
    """Bull Call Spread analysis using real chain data."""
    low_strike: float
    high_strike: float
    long_leg: ChainOption
    short_leg: ChainOption
    long_call_price: float
    short_call_price: float
    net_debit: float
    max_profit: float
    max_loss: float
    breakeven: float
    risk_reward_ratio: float
    bid_ask_slippage: float       # Total friction vs mid-market
    long_greeks: Dict
    short_greeks: Dict
    net_greeks: Dict


def analyze_spread_from_chain(spot_price: float, chain: list[ChainOption], 
                              strategy: 'BullCallSpreadStrategy') -> ChainSpreadAnalysis:
    """Analyze bull call spread using real chain data with bid-ask slippage."""
    if not chain:
        raise ValueError('chain cannot be empty')
    
    T = strategy.days_to_expiry / 365.0
    
    # Filter by liquidity
    liquid_chain = filter_by_liquidity(chain, min_volume=5, min_open_interest=50)
    if not liquid_chain:
        liquid_chain = chain  # fallback if nothing passes filter
    
    # Select strikes
    strikes = sorted(set(o.strike for o in liquid_chain))
    target_low = spot_price * (1 + strategy.low_strike_offset)
    target_high = spot_price * (1 + strategy.high_strike_offset)
    
    low_strike = select_strike(target_low, strikes)
    high_strike = select_strike(target_high, strikes)
    
    # Find legs
    long_leg = next((o for o in liquid_chain if o.strike == low_strike), None)
    short_leg = next((o for o in liquid_chain if o.strike == high_strike), None)
    
    if not long_leg or not short_leg:
        raise ValueError(f'Could not find chain rows. Low={low_strike}, High={high_strike}')
    
    # Realistic fills: buy long at ask, sell short at bid
    long_call_price = long_leg.ask
    short_call_price = short_leg.bid
    
    # Greeks with per-strike IV (captures vol skew)
    lg = crr_greeks(spot_price, low_strike, T, strategy.risk_free_rate, long_leg.iv, 'call')
    sg = crr_greeks(spot_price, high_strike, T, strategy.risk_free_rate, short_leg.iv, 'call')
    
    net_debit = (long_call_price - short_call_price) * strategy.contracts * 100
    spread_width = (high_strike - low_strike) * strategy.contracts * 100
    max_profit = spread_width - net_debit
    max_loss = net_debit
    breakeven = low_strike + net_debit / (strategy.contracts * 100)
    risk_reward = max_profit / max_loss if max_loss > 0 else 0
    
    # Bid-ask slippage vs mid-market
    bid_ask_slippage = ((long_leg.ask - long_leg.mid) +
                        (short_leg.mid - short_leg.bid)) * strategy.contracts * 100
    
    net_greeks = {k: lg[k] - sg[k] for k in lg}
    
    return ChainSpreadAnalysis(
        low_strike=low_strike,
        high_strike=high_strike,
        long_leg=long_leg,
        short_leg=short_leg,
        long_call_price=long_call_price,
        short_call_price=short_call_price,
        net_debit=net_debit,
        max_profit=max_profit,
        max_loss=max_loss,
        breakeven=breakeven,
        risk_reward_ratio=risk_reward,
        bid_ask_slippage=bid_ask_slippage,
        long_greeks=lg,
        short_greeks=sg,
        net_greeks=net_greeks,
    )


@dataclass
class ExecutabilityAssessment:
    """Score and assess whether a spread is executable."""
    score: int                    # 0-100
    execute_now: bool
    warnings: list[str]
    spread_spread_pct: float
    long_spread_pct: float
    short_spread_pct: float


def assess_executability(analysis: ChainSpreadAnalysis) -> ExecutabilityAssessment:
    """Assess whether the spread is executable based on liquidity and spreads."""
    warnings = []
    score = 100
    
    long_spread_pct = bid_ask_spread_pct(analysis.long_leg)
    short_spread_pct = bid_ask_spread_pct(analysis.short_leg)
    spread_spread_pct = (abs(analysis.bid_ask_slippage) / analysis.max_loss * 100) if analysis.max_loss > 0 else 0
    
    # Check volume
    if analysis.long_leg.volume < 50:
        warnings.append('Long leg: low volume')
        score -= 10
    if analysis.long_leg.volume < 10:
        warnings.append('Long leg: very low volume')
        score -= 20
    
    if analysis.short_leg.volume < 50:
        warnings.append('Short leg: low volume')
        score -= 10
    if analysis.short_leg.volume < 10:
        warnings.append('Short leg: very low volume')
        score -= 20
    
    # Check open interest
    if analysis.long_leg.open_interest < 100:
        warnings.append('Long leg: thin OI')
        score -= 5
    if analysis.short_leg.open_interest < 100:
        warnings.append('Short leg: thin OI')
        score -= 5
    
    # Check bid-ask relative to spread width
    if spread_spread_pct > 10:
        warnings.append(f'Bid-ask is {spread_spread_pct:.1f}% of max loss')
        score -= 15
    if spread_spread_pct > 20:
        warnings.append('Slippage risk is high')
        score -= 25
    
    # Check individual leg spreads
    if long_spread_pct > 5:
        warnings.append(f'Long leg spread: {long_spread_pct:.1f}%')
        score -= 5
    if short_spread_pct > 5:
        warnings.append(f'Short leg spread: {short_spread_pct:.1f}%')
        score -= 5
    
    score = max(0, min(100, score))
    
    return ExecutabilityAssessment(
        score=score,
        execute_now=score >= 70,
        warnings=warnings,
        spread_spread_pct=spread_spread_pct,
        long_spread_pct=long_spread_pct,
        short_spread_pct=short_spread_pct,
    )


@dataclass
class SpreadCostComparison:
    """Compare theoretical vs realistic execution cost."""
    theoretical_debit: float
    realistic_debit: float
    slippage_dollars: float
    slippage_pct: float
    contracts: int


def compare_costs(analysis: ChainSpreadAnalysis) -> SpreadCostComparison:
    """Show actual cost penalty of execution vs mid-market."""
    theoretical_debit = (analysis.long_leg.mid - analysis.short_leg.mid) * 100
    realistic_debit = analysis.net_debit
    slippage = realistic_debit - theoretical_debit
    slippage_pct = (slippage / theoretical_debit * 100) if theoretical_debit > 0 else 0
    
    return SpreadCostComparison(
        theoretical_debit=theoretical_debit,
        realistic_debit=realistic_debit,
        slippage_dollars=slippage,
        slippage_pct=slippage_pct,
        contracts=1,  # or pass from strategy
    )


if __name__ == "__main__":
    strategy = BullCallSpreadStrategy(low_strike_offset=0.0, high_strike_offset=0.05, days_to_expiry=30)
    result = strategy.evaluate(spot_price=150.0, historical_vol=0.25)
    print("Bull Call Spread Analysis (CRR Pricing)")
    print("=" * 45)
    print(f"Long Call (CRR):   ${result.long_call_price:.4f}")
    print(f"Short Call (CRR):  ${result.short_call_price:.4f}")
    print(f"Net Debit:         ${result.net_debit:.2f}")
    print(f"Max Profit:        ${result.max_profit:.2f}")
    print(f"Max Loss:          ${result.max_loss:.2f}")
    print(f"Breakeven:         ${result.breakeven:.2f}")
    print(f"Risk/Reward:       {result.risk_reward_ratio:.2f}x")
    greeks = strategy.greeks(150.0, 0.25)
    print(f"\nNet Greeks: {greeks}")
