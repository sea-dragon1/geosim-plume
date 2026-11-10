# E03 checkpoint-compatible window protocol (2026-09-25)

This supersedes the earlier full-time GNN call for the existing checkpoints.
The original full-time GNN outputs remain archived and must not be used.

The inherited decoder indexes its time-factor base as step + Q - 1.
In the trained/evaluated 21-frame canonical sample (step=Q=10), its indices
clamp to frame20 and all ten factors equal t20-t19. Supplying a longer time
axis changed this operator immediately, even at the first predicted window.
This was confirmed by GPU job 3585279.

The compatibility wrapper now invokes that same learned window operator at
each block: input states [s-Q:s], output states [s:s+Q], and time values
through index s+Q. Absolute graph time conditioning stays at t[s].
One scheduled timestamp beyond the output block is required. These timestamps
are exogenous; no future physical states enter autonomous prediction.
The wrapper changes neither checkpoint weights nor the inherited decoder.
It does not claim to implement the intended cumulative physical increments.
That different model would require retraining and separate evaluation.

GPU acceptance per checkpoint:
- All-test first-window score vs E02, relative tolerance 0.001.
- Every teacher block vs the ORIGINAL model on an independent 21-frame sample.
- First auto/teacher block agrees.
- Zeroing future state values leaves autonomous predictions unchanged.
- Changing timestamps after the local 21-frame boundary leaves the first
  predicted block unchanged.
- Full formal evaluation and recalibrated validation thresholds follow only
  after those checks pass.

Eight corrected graph configurations: formal_gnn_compat_20260925.
Other validated results: formal_20260925 (FNO/Transolver only) and
formal_gnot_20260925 (GNOT).
Final 20-checkpoint, five-model acceptance: formal_all5_20260925.
Do not interpret failed range/error thresholds as NaN/Inf or numerical blow-up.
