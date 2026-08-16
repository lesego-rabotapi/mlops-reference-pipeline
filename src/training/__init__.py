"""
Package marker for training and evaluation.

Why it exists: makes the training entrypoint importable and executable as a module.
Responsible for: declaring the training namespace only.
Must not: train models or write artifacts on import.
"""
