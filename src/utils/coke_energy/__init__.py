"""BF-02 coke energy model: combined mass + energy balance, fixed-eta CO and two-zone (Rist) coke-rate models.

Provenance. The model was built in the Claude web app (14-15 Oct 2026) and delivered as BF02_model_code.zip (newest
bundle, 15 Oct 01:14; copy under tmp/claude_web_analysis/). By decision of 15 Oct it is kept almost unchanged here, so
the validated numbers (results write-up "BF-02 coke model: what the latest results mean") stay reproducible:

    bf_thermo.py        verbatim
    bf_balance.py       verbatim except the import path
    rist.py             verbatim except import paths
    run_shifts.py       functions verbatim; script ``__main__`` moved to pipeline.py
    whatif.py           verbatim except import paths
    burnt.py            functions verbatim; ``__main__`` moved to pipeline.py; module-level warnings filter removed
    twozone_shifts.py   ``calibrate_all`` verbatim; ``__main__`` moved to pipeline.py; module-level warnings filter removed
    build_shifts.py     script body wrapped in ``build_shift_inputs(fd, m10, hm, dpr)``; statements unchanged
    rolling_predict.py  script body wrapped in ``rolling_predict(S, R)``; statements unchanged

New in the app (no change to the model):

    pipeline.py         the scripts' ``__main__`` blocks run in memory, plus one joined per-shift table
    rist_diagram.py     Rist operating-diagram geometry from the two-zone model's own quantities

Pure numerical code: no Streamlit and no I/O. Data loaders are in ``data.coke_energy.sources``.
"""
