"""
Bull Call Spread Strategy
Defined-risk options strategy using two call options at different strikes.

Structure: Buy lower-strike call, sell higher-strike call (same expiry)
Max Profit: (high_strike - low_strike - net_debit) * 100
Max Loss:   net_debit * 100

Author: Agentic Trading
Version: 1.0.0
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, date


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


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Price a European call option using Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict:
    """Calculate option Greeks."""
    if T <= 0 or sigma <= 0:
        return {'delta': 1.0 if S > K else 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}


class BullCallSpreadStrategy:
    """
    Bull Call Spread Strategy

    A defined-risk, defined-reward options strategy for moderately bullish
    outlooks. Consists of buying an ATM/slightly OTM call and selling a
    further OTM call to reduce the net cost.

    Parameters:
    -----------
    low_strike_offset : float
        Low strike as % above/below current price (default: 0.0 = ATM)
    high_strike_offset : float
        High strike as % above current price (default: 0.05 = 5% OTM)
    days_to_expiry : int
        Days until expiry (default: 30)
    risk_free_rate : float
        Annual risk-free rate (default: 0.05)
    implied_volatility : float
        Implied volatility — override if known (default: None, uses historical)
    contracts : int
        Number of contracts to trade (default: 1)

    Example:
    --------
    >>> strategy = BullCallSpreadStrategy(low_strike_offset=0.0, high_strike_offset=0.05)
    >>> result = strategy.evaluate(spot_price=150.0, historical_vol=0.25)
    >>> print(f"Max Profit: ${result.max_profit:.2f}")
    >>> print(f"Breakeven:  ${result.breakeven:.2f}")
    """

    def __init__(
        self,
        low_strike_offset: float = 0.0,
        high_strike_offset: float = 0.05,
        days_to_expiry: int = 30,
        risk_free_rate: float = 0.05,
        implied_volatility: Optional[float] = None,
        contracts: int = 1,
    ):
        self.low_strike_offset = low_strike_offset
        self.high_strike_offset = high_strike_offset
        self.days_to_expiry = days_to_expiry
        self.risk_free_rate = risk_free_rate
        self.implied_volatility = implied_volatility
        self.contracts = contracts

    def evaluate(
        self,
        spot_price: float,
        historical_vol: float = 0.25,
        target_price: Optional[float] = None,
    ) -> SpreadResult:
        """
        Evaluate the spread at current spot price.

        Parameters:
        -----------
        spot_price : float
            Current underlying price
        historical_vol : float
            Historical volatility (used if implied_volatility not set)
        target_price : float, optional
            Expected price at expiry for profit calculation

        Returns:
        --------
        SpreadResult with all spread metrics
        """
        sigma = self.implied_volatility or historical_vol
        T = self.days_to_expiry / 365.0
        r = self.risk_free_rate

        low_strike = round(spot_price * (1 + self.low_strike_offset), 2)
        high_strike = round(spot_price * (1 + self.high_strike_offset), 2)

        long_call = black_scholes_call(spot_price, low_strike, T, r, sigma)
        short_call = black_scholes_call(spot_price, high_strike, T, r, sigma)

        net_debit = (long_call - short_call) * self.contracts * 100
        spread_width = (high_strike - low_strike) * self.contracts * 100
        max_profit = spread_width - net_debit
        max_loss = net_debit
        breakeven = low_strike + (net_debit / (self.contracts * 100))
        risk_reward = max_profit / max_loss if max_loss > 0 else 0

        # Profit at a specific target price
        if target_price is None:
            target_price = high_strike
        long_pnl = max(target_price - low_strike, 0) - long_call
        short_pnl = short_call - max(target_price - high_strike, 0)
        profit_at_target = (long_pnl + short_pnl) * self.contracts * 100

        return SpreadResult(
            long_call_price=long_call,
            short_call_price=short_call,
            net_debit=net_debit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven=breakeven,
            risk_reward_ratio=risk_reward,
            profit_at_expiry=profit_at_target,
        )

    def greeks(self, spot_price: float, historical_vol: float = 0.25) -> Dict:
        """Net Greeks of the spread position."""
        sigma = self.implied_volatility or historical_vol
        T = self.days_to_expiry / 365.0
        r = self.risk_free_rate

        low_strike = spot_price * (1 + self.low_strike_offset)
        high_strike = spot_price * (1 + self.high_strike_offset)

        long_greeks = black_scholes_greeks(spot_price, low_strike, T, r, sigma)
        short_greeks = black_scholes_greeks(spot_price, high_strike, T, r, sigma)

        return {
            'net_delta': long_greeks['delta'] - short_greeks['delta'],
            'net_gamma': long_greeks['gamma'] - short_greeks['gamma'],
            'net_theta': long_greeks['theta'] - short_greeks['theta'],
            'net_vega': long_greeks['vega'] - short_greeks['vega'],
        }

    def pnl_at_expiry(self, spot_price: float, price_range: Tuple[float, float], steps: int = 50) -> Dict:
        """Calculate P&L across a range of underlying prices at expiry."""
        low_strike = spot_price * (1 + self.low_strike_offset)
        high_strike = spot_price * (1 + self.high_strike_offset)
        long_cost = black_scholes_call(spot_price, low_strike, self.days_to_expiry / 365, self.risk_free_rate,
                                       self.implied_volatility or 0.25)
        short_credit = black_scholes_call(spot_price, high_strike, self.days_to_expiry / 365, self.risk_free_rate,
                                          self.implied_volatility or 0.25)
        net_debit = long_cost - short_credit

        prices = np.linspace(price_range[0], price_range[1], steps)
        pnls = []
        for p in prices:
            long_val = max(p - low_strike, 0)
            short_val = max(p - high_strike, 0)
            pnl = (long_val - short_val - net_debit) * self.contracts * 100
            pnls.append(pnl)

        return {'prices': prices.tolist(), 'pnl': pnls}


if __name__ == "__main__":
    strategy = BullCallSpreadStrategy(
        low_strike_offset=0.0,
        high_strike_offset=0.05,
        days_to_expiry=30,
    )
    result = strategy.evaluate(spot_price=150.0, historical_vol=0.25)
    print("Bull Call Spread Analysis")
    print("=" * 40)
    print(f"Net Debit:         ${result.net_debit:.2f}")
    print(f"Max Profit:        ${result.max_profit:.2f}")
    print(f"Max Loss:          ${result.max_loss:.2f}")
    print(f"Breakeven:         ${result.breakeven:.2f}")
    print(f"Risk/Reward:       {result.risk_reward_ratio:.2f}x")
