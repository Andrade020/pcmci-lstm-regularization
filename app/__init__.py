"""Causal forecasting app: plug-and-play front-end + honest benchmark harness.

Built entirely on top of the paper's `src/` library (PCMCI causal discovery +
causal-regularized LSTM). Nothing here reimplements the modelling pipeline; the
sub-app is a thin, reusable orchestration layer around it.
"""
