# Decision memo: replenishment policy for weekly store orders

**Question.** How should weekly order quantities be set for 30,490 item-store
combinations (Walmart M5 data, 10 stores), and how should a limited stock budget be
spread across items?

**Recommendation.** Set order-up-to levels from quantile forecasts of demand over the
lead time plus review period, choose the quantile by simulated cost rather than by the
textbook critical ratio, and when budget is tight allocate it with the LP rather than
cutting every item by the same percentage. Before rolling this out, decide whether the
budget should maximise units or revenue, and validate the cost assumptions.

**Evidence** (backtest February-May 2016; held-out period April 23 - May 20, 2016).

| Decision | Compared with | Result |
|---|---|---|
| Forecasting model | 4-week moving average (best simple method) | LightGBM error (WAPE) 0.353 vs 0.379, consistent across all 4 test periods |
| Order-up-to policy, service level tuned on cost | Moving average + 1 week safety stock | 9.8% lower total cost (95% CI 9.0-10.7%) |
| Same | Forecast mean + normal safety stock | 0.9% lower cost (CI 0.3-1.4%); 12-15% lower when lost sales cost 49x holding |
| Budget at 50% of full stocking | Cut every item proportionally | 26.7% more units sold, but about 9% less revenue |

**What would change the recommendation.**

- The textbook rule (service level = lost-sale cost / (lost-sale + holding cost))
  made the quantile policy 4.3% *more* expensive than the normal policy in the main
  case. The gain depends on tuning the service level; the tuned level was 0.8 where the
  rule says 0.9.
- If the business goal is revenue or margin, the allocation should optimise that
  instead; the current LP favours cheap, fast-selling items.

**Assumptions and caveats.** Costs (holding 25% of price per year; lost sale 9x the
weekly holding cost in the main case), a 2-week lead time and weekly review are
assumptions, not Walmart data. Sales stand in for demand, which understates demand
when shelves were empty. The test period has no holiday season, and the held-out period
is only 4 weeks.
