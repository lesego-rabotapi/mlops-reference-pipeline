"""
Package marker for the validation stage.

Why it exists: makes validation modules importable as a stable package.
Responsible for: declaring the validation namespace only.
Must not: execute validation or publish datasets on import.
"""
