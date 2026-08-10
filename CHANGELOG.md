# Changelog

## 0.6.1

- Add configurable neutralization power to `generate_neutralized_weights`,
  `alpha`, and `meta_portfolio_contribution`; the new default is `1`, while
  callers can pass `power=1.5` to reproduce historical Alpha/MPC behavior.
- Preserve the legacy Signals turnover definition with an explicit default
  power of `1.5`.
