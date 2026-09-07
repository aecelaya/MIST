"""Shared fixture/pipeline-running test harness, originally for the ANTs ->
SimpleITK migration (now complete -- MIST no longer depends on ants at all).

Built to run a small, deterministic set of edge-case fixtures through
``mist_analyze -> mist_preprocess`` and capture artifacts that later migration
stages diffed against to catch silent, non-crashing (spatially
transposed/flipped) bugs. That migration-specific diffing usage is gone, but
:mod:`fixtures` (``generate_dataset``) and :mod:`harness` (``run_pipeline``)
are still real, load-bearing infrastructure -- reused by
``tests/data_loading/test_generic_loader.py`` and
``tests/regression/cpu_rocm/`` for real end-to-end pipeline tests. See the
README in this package for details.
"""
