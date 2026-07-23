"""Stage 0 regression harness for the ANTs -> SimpleITK migration.

See ``ants_to_simpleitk_migration.md`` (Stage 0) and the README in this
package for the intent: build a small, deterministic set of edge-case
fixtures, run the *current* pipeline over them, and capture golden artifacts
that later migration stages diff against to catch silent, non-crashing
(spatially transposed/flipped) bugs.
"""
