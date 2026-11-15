"""Synthetic datasets and a real analyze -> preprocess runner for tests.

:mod:`fixtures` (``generate_dataset``) writes a small, deterministic set of
edge-case images and labels. :mod:`harness` (``run_pipeline``) runs the real
``mist_analyze -> mist_preprocess`` on them, for tests that need real
preprocessed data rather than mocked pipeline internals.
"""
