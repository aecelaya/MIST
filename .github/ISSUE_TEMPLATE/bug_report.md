---
name: Bug report
about: Create a report to help us improve
title: ""
labels: bug
assignees: ""
---

**Describe the bug** A clear and concise description of what the bug is.

**To Reproduce** Steps to reproduce the behavior, including the exact command(s)
you ran:

```
# e.g. mist_run_all --data dataset.json --numpy ./numpy --results ./results ...
```

**Expected behavior** A clear and concise description of what you expected to
happen.

**Screenshots / logs** If applicable, add screenshots or paste the full error
traceback (not just the last line) to help explain your problem.

**MIST installation**

- Install method: [pip / Docker / from source]
- MIST version: [output of `pip show mist-medical`, or the Docker image tag,
  e.g. mistmedical/mist:2.0.1rc0]

**Hardware/software stack (please complete the following information):**

- OS: [i.e., Ubuntu 22.04]
- Number of GPUs and multi-GPU (DDP)?: [i.e., 4x GPUs, multi-GPU]
- GPU: [i.e., A100]
- Drivers: [i.e., CUDA 12.4]
- PyTorch: [i.e., 2.9.1 — from
  `python -c "import torch; print(torch.__version__)"`]

**Additional context** Add any other context about the problem here.
