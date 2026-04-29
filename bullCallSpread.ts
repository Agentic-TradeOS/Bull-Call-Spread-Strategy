/**
 * Bull Call Spread Strategy
 * Defined-risk options strategy: buy lower-strike call, sell higher-strike call.
 *
 * Pricing: Cox-Ross-Rubinstein (CRR) binomial tree — American-style exercise.
 * This is more accurate than Black-Scholes for US equity options because it
 * correctly models early-exercise value via backward induction.
 *
 * Max Profit : (highStrike − lowStrike − netDebit) × contracts × 100
 * Max Loss   : netDebit × contracts × 100
 *
 * Author : Agentic Trading
 * Version: 2.0.0
 */

// ─── CRR Binomial Tree ────────────────────────────────────────────────────────

export type OptionType  = 'call' | 'put';
export type OptionStyle = 'american' | 'european';

export interface CRRInputs {
  spotPrice:     number;
  strikePrice:   number;
  timeToExpiry:  number;   // years
  riskFreeRate:  number;   // annual, e.g. 0.05
  volatility:    number;   // annual, e.g. 0.25
  optionType:    OptionType;
  optionStyle?:  OptionStyle;  // default 'american'
  steps?:        number;       // default 100
}

/**
 * Price an option using the Cox-Ross-Rubinstein binomial tree.
 *
 * Steps:
 *  1. Build terminal stock prices:  S · u^j · d^(n−j)  for j = 0…n
 *  2. Calculate terminal payoffs
 *  3. Backward-induct discounting each node by e^(−r·Δt)
 *  4. For American options, compare intrinsic value at every node
 */
export function crrPrice(inputs: CRRInputs): number {
  const {
    spotPrice:    S,
    strikePrice:  K,
    timeToExpiry: T,
    riskFreeRate: r,
    volatility:   sigma,
    optionType,
    optionStyle = 'american',
    steps: n    = 100,
  } = inputs;

  if (T <= 0 || sigma <= 0) {
    return optionType === 'call' ? Math.max(S - K, 0) : Math.max(K - S, 0);
  }

  const dt   = T / n;
  const u    = Math.exp(sigma * Math.sqrt(dt));   // up factor
  const d    = 1 / u;                              // down factor (recombining)
  const disc = Math.exp(-r * dt);                  // per-step discount
  const p    = (Math.exp(r * dt) - d) / (u - d);  // risk-neutral up probability

  // Terminal payoffs
  const values: number[] = Array.from({ length: n + 1 }, (_, j) => {
    const spotT = S * Math.pow(u, j) * Math.pow(d, n - j);
    return optionType === 'call' ? Math.max(spotT - K, 0) : Math.max(K - spotT, 0);
  });

  // Backward induction
  for (let i = n - 1; i >= 0; i--) {
    for (let j = 0; j <= i; j++) {
      const continuation = disc * (p * values[j + 1] + (1 - p) * values[j]);
      if (optionStyle === 'american') {
        const spotIJ    = S * Math.pow(u, j) * Math.pow(d, i - j);
        const intrinsic = optionType === 'call'
          ? Math.max(spotIJ - K, 0)
          : Math.max(K - spotIJ, 0);
        values[j] = Math.max(intrinsic, continuation);
      } else {
        values[j] = continuation;
      }
    }
  }

  return Math.round(values[0] * 1e6) / 1e6;
}

/** Central finite-difference Greeks from the CRR tree. */
export interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega:  number;
  rho:   number;
}

export function crrGreeks(inputs: CRRInputs): Greeks {
  const { spotPrice: S, volatility: sigma, riskFreeRate: r, timeToExpiry: T } = inputs;
  const dS = S * 0.01, dsig = 0.01, dr = 0.005, dT = 1 / 365;

  const base = crrPrice(inputs);
  const pu   = crrPrice({ ...inputs, spotPrice: S + dS });
  const pd   = crrPrice({ ...inputs, spotPrice: S - dS });
  const pvu  = crrPrice({ ...inputs, volatility: sigma + dsig });
  const pvd  = crrPrice({ ...inputs, volatility: Math.max(sigma - dsig, 0.01) });
  const pru  = crrPrice({ ...inputs, riskFreeRate: r + dr });
  const prd  = crrPrice({ ...inputs, riskFreeRate: Math.max(r - dr, 0.001) });
  const pt   = T > dT ? crrPrice({ ...inputs, timeToExpiry: T - dT }) : base;

  return {
    delta: Math.round(((pu - pd) / (2 * dS))          * 1e4) / 1e4,
    gamma: Math.round(((pu - 2 * base + pd) / dS ** 2) * 1e4) / 1e4,
    theta: Math.round(((pt - base) / dT / 365)         * 1e4) / 1e4,
    vega:  Math.round(((pvu - pvd) / (2 * dsig) / 100) * 1e4) / 1e4,
    rho:   Math.round(((pru - prd) / (2 * dr)  / 100)  * 1e4) / 1e4,
  };
}

// ─── Spread config & analysis ─────────────────────────────────────────────────

export interface SpreadConfig {
  lowStrikeOffset:    number;   // % offset for long call  (0.0 = ATM)
  highStrikeOffset:   number;   // % offset for short call (0.05 = 5% OTM)
  daysToExpiry:       number;
  riskFreeRate:       number;
  impliedVolatility?: number;
  contracts:          number;
  crrSteps?:          number;   // default 100
}

export const defaultConfig: SpreadConfig = {
  lowStrikeOffset:  0.0,
  highStrikeOffset: 0.05,
  daysToExpiry:     30,
  riskFreeRate:     0.05,
  contracts:        1,
  crrSteps:         100,
};

export interface SpreadAnalysis {
  lowStrike:      number;
  highStrike:     number;
  longCallPrice:  number;
  shortCallPrice: number;
  netDebit:       number;
  maxProfit:      number;
  maxLoss:        number;
  breakeven:      number;
  riskReward:     number;
  longGreeks:     Greeks;
  shortGreeks:    Greeks;
  netGreeks:      Greeks;
}

export function analyzeSpread(
  spotPrice: number,
  vol: number,
  config: SpreadConfig = defaultConfig,
): SpreadAnalysis {
  const sigma    = config.impliedVolatility ?? vol;
  const T        = config.daysToExpiry / 365;
  const { riskFreeRate: r, contracts, crrSteps: steps = 100 } = config;

  const lowStrike  = Math.round(spotPrice * (1 + config.lowStrikeOffset)  * 100) / 100;
  const highStrike = Math.round(spotPrice * (1 + config.highStrikeOffset) * 100) / 100;

  const base = (strikePrice: number): CRRInputs => ({
    spotPrice, strikePrice, timeToExpiry: T, riskFreeRate: r, volatility: sigma,
    optionType: 'call', optionStyle: 'american', steps,
  });

  const longCallPrice  = crrPrice(base(lowStrike));
  const shortCallPrice = crrPrice(base(highStrike));
  const longGreeks     = crrGreeks(base(lowStrike));
  const shortGreeks    = crrGreeks(base(highStrike));

  const netDebit    = (longCallPrice - shortCallPrice) * contracts * 100;
  const spreadWidth = (highStrike - lowStrike) * contracts * 100;
  const maxProfit   = spreadWidth - netDebit;
  const maxLoss     = netDebit;
  const breakeven   = lowStrike + netDebit / (contracts * 100);
  const riskReward  = maxLoss > 0 ? maxProfit / maxLoss : 0;

  const netGreeks: Greeks = {
    delta: longGreeks.delta - shortGreeks.delta,
    gamma: longGreeks.gamma - shortGreeks.gamma,
    theta: longGreeks.theta - shortGreeks.theta,
    vega:  longGreeks.vega  - shortGreeks.vega,
    rho:   longGreeks.rho   - shortGreeks.rho,
  };

  return {
    lowStrike, highStrike, longCallPrice, shortCallPrice,
    netDebit, maxProfit, maxLoss, breakeven, riskReward,
    longGreeks, shortGreeks, netGreeks,
  };
}

export function pnlAtExpiry(
  spotPrice: number,
  vol: number,
  priceRange: [number, number],
  steps = 50,
  config: SpreadConfig = defaultConfig,
): Array<{ price: number; pnl: number }> {
  const { lowStrike, highStrike, longCallPrice, shortCallPrice } = analyzeSpread(spotPrice, vol, config);
  const netDebit = longCallPrice - shortCallPrice;
  const step     = (priceRange[1] - priceRange[0]) / steps;

  return Array.from({ length: steps + 1 }, (_, i) => {
    const price = priceRange[0] + i * step;
    const pnl   = (Math.max(price - lowStrike, 0) - Math.max(price - highStrike, 0) - netDebit)
                  * config.contracts * 100;
    return { price, pnl };
  });
}
