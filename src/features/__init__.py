"""
Package marker for the feature-engineering stage.

Why it exists: makes feature modules importable as a stable package.
Responsible for: declaring the feature-engineering namespace only.
Must not: fit preprocessors or write artifacts on import.
"""
