## Bull Call Spread Calculator
A lightweight implementation of the **Bull Call Spread** options strategy. This repository provides core logic in both **Python** and **TypeScript** to calculate the risk-to-reward profile of a vertical debit spread.
### Strategy Overview
The Bull Call Spread is used when you anticipate a moderate increase in the underlying asset's price. By purchasing a call option and selling another at a higher strike price, you offset the cost of the trade.
 * **Direction:** Bullish
 * **Risk:** Limited to the **Net Debit** paid.
 * **Reward:** Capped at the difference between strikes minus the net debit.
### Code Samples
#### 🐍 Python
Ideal for data analysis or backtesting scripts.
```python
long_p, short_p = 5.00, 2.00
long_s, short_s = 100, 105

net_debit = long_p - short_p
max_profit = (short_s - long_s) - net_debit

print(f"Max Risk: ${net_debit} | Max Profit: ${max_profit}")

```
#### 🟦 TypeScript
Ready for integration into web-based trading dashboards.
```typescript
const calculateSpread = (lp: number, sp: number, ls: number, ss: number) => {
  const netDebit = lp - sp;
  const maxProfit = (ss - ls) - netDebit;
  return { netDebit, maxProfit };
};

```
### Quick Reference
| Metric | Formula |
|---|---|
| **Net Debit** | Cost_{Long} - Credit_{Short} |
| **Max Profit** | (Strike_{Short} - Strike_{Long}) - NetDebit |
| **Breakeven** | Strike_{Long} + NetDebit |
