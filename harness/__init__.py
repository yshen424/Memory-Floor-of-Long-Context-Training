"""Prediction harness: configuration -> each mechanism's peak-memory line.

config.py holds the shapes and platforms, terms.py the byte model, mechanisms.py
the lines. Validation: tests/test_harness_v100.py.
"""
from .config import ModelShape, Run, Platform, PLATFORMS, TIER_A, LLAMA32_3B, V100_CAP_DELTA_MIB
from .terms import param_counts, slope_terms, intercept_terms, floor_extra_terms
from .mechanisms import Line, static_line, partial_line, floor_line, line_for, peak
from .capacity import CapacityModel, calibrate, bracket, request_mib_per_x, capped_model
from .timemodel import TimeCalibration, fit_step_time, fit_kappa
from .report import predict, header_block, write_json
