#!/usr/bin/env python
"""
Script to generate example plots for a run with nessai
"""

from nessai_models import Gaussian

from nessai.flowsampler import FlowSampler

output_dir = "example_run"
model = Gaussian()

fs = FlowSampler(
    model, nlive=500, output=output_dir, checkpointing=False, resume=False
)
fs.run()
