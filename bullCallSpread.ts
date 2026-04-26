export interface SpreadConfig {
  lowStrikeOffset: number;
  highStrikeOffset: number;
  daysToExpiry: number;
  riskFreeRate: number;
  impliedVolatility?: number;
  contracts: number;
}

export const defaultConfig: SpreadConfig = {
  lowStrikeOffset: 0.0,
  highStrikeOffset: 0.05,
  daysToExpiry: 30,
  riskFreeRate: 0.05,
  contracts: 1,
};

function normalCDF(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1.0 + sign * y);
}

export function blackScholesCall(S: number, K: number, T: number, r: number, sigma: number): number {
  if (T <= 0 || sigma <= 0) return Math.max(S - K, 0);
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  return S * normalCDF(d1) - K * Math.exp(-r * T) * normalCDF(d2);
}

export interface SpreadAnalysis {
  lowStrike: number;
  highStrike: number;
  longCallPrice: number;
  shortCallPrice: number;
  netDebit: number;
  maxProfit: number;
  maxLoss: number;
  breakeven: number;
  riskReward: number;
}

export function analyzeSpread(spotPrice: number, vol: number, config: SpreadConfig = defaultConfig): SpreadAnalysis {
  const sigma = config.impliedVolatility ?? vol;
  const T = config.daysToExpiry / 365;
  const { riskFreeRate: r, contracts } = config;

  const lowStrike = Math.round(spotPrice * (1 + config.lowStrikeOffset) * 100) / 100;
  const highStrike = Math.round(spotPrice * (1 + config.highStrikeOffset) * 100) / 100;

  const longCall = blackScholesCall(spotPrice, lowStrike, T, r, sigma);
  const shortCall = blackScholesCall(spotPrice, highStrike, T, r, sigma);

  const netDebit = (longCall - shortCall) * contracts * 100;
  const spreadWidth = (highStrike - lowStrike) * contracts * 100;
  const maxProfit = spreadWidth - netDebit;
  const maxLoss = netDebit;
  const breakeven = lowStrike + netDebit / (contracts * 100);
  const riskReward = maxLoss > 0 ? maxProfit / maxLoss : 0;

  return { lowStrike, highStrike, longCallPrice: longCall, shortCallPrice: shortCall, netDebit, maxProfit, maxLoss, breakeven, riskReward };
}

export function pnlAtExpiry(
  spotPrice: number,
  vol: number,
  priceRange: [number, number],
  steps = 50,
  config: SpreadConfig = defaultConfig
): Array<{ price: number; pnl: number }> {
  const { lowStrike, highStrike, longCallPrice, shortCallPrice } = analyzeSpread(spotPrice, vol, config);
  const netDebit = longCallPrice - shortCallPrice;
  const step = (priceRange[1] - priceRange[0]) / steps;

  return Array.from({ length: steps + 1 }, (_, i) => {
    const price = priceRange[0] + i * step;
    const longVal = Math.max(price - lowStrike, 0);
    const shortVal = Math.max(price - highStrike, 0);
    const pnl = (longVal - shortVal - netDebit) * config.contracts * 100;
    return { price, pnl };
  });
}
