# alphalens_trade_generation
blends NeuralProphet/N-HiTS/Prophet/TFT mean models with a production-hardened GARCH(1,1) Student-t volatility engine. It covers training, evaluation, rolling inference, Monte Carlo, and TP/SL sizing. Diagnostics rely on conditional_volatility vs |returns|, while rolling sigma feeds the live risk engine to produce ready-to-use trade payloads.
