"""WBC (stage-2 trajectory tracking) benchmark mode.

Reuses the shared infrastructure from ``benchmark.dog_policy.evaluation``
(Accumulator, ScenarioResult, HTML reports, comparison tool) and extends it
with WBC-specific metrics: EE tracking error, reach utilisation, energy,
and smoothness (acceleration / jerk).

Layout
------
``evaluation``     env/policy loading, extended Accumulator, eval loop
``scenarios``      WBC scenario definitions (bank-trajectory aggregate,
                   bandwidth probe, workspace probe)
"""
