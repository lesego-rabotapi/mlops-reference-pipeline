"""
Package marker for explicit pipeline configuration.

Why it exists: makes configuration importable from every execution context.
Responsible for: grouping configuration modules only.
Must not: create directories, run stages, or load artifacts.
"""
