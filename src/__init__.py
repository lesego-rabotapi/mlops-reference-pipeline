"""
Package marker for the local MLOps pipeline.

Why it exists: enables reliable imports after an editable installation.
Responsible for: defining the source package boundary only.
Must not: run pipeline stages or hold runtime configuration.
"""
