# Benchmarks

Manual checks of MIST on real GPUs. These aren't tests: nothing here runs in CI
or passes or fails on its own. Each produces numbers or files to read, and some
of MIST's constants (the planner's memory estimate, for example) come from them.
For MIST demos, see [`examples/`](../examples/).

Run these on a machine with a GPU and MIST installed. Each script's docstring
gives its command line.

| Script | What it checks | When to run it |
| --- | --- | --- |
| [`calibrate_planning.py`](calibrate_planning.py) | The planner's memory estimate against peak memory, for every network | After changing a network or the planner's constants, once per GPU |
| [`check_training.py`](check_training.py) | That the default pipeline analyzes, preprocesses, trains, and predicts on a real dataset | Before a release |
| [`check_augmentation.py`](check_augmentation.py) | Training augmentation on real preprocessed data, with patches saved to inspect | After changing augmentation |
