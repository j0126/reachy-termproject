"""Phase 1 analysis: reproductions of SDK 1.11.0 behaviour computed with the SDK's own code.

Everything except ``sim_smoke`` is *computed* (pure functions of the SDK source, no daemon);
results written by ``reachy-motion phase1`` carry ``"data_kind": "computed"``. ``sim_smoke``
drives a local MuJoCo daemon spawned by ``backends.sim_daemon`` and writes ``"simulated"`` data.
Nothing in this package talks to the real robot.
"""
