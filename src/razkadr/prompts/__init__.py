"""Versioned provider prompts packaged with the library (naryad 008).

Files are named ``<purpose>.v<N>.md``; the version in the filename is
part of the stage contract - a prompt change that could alter provider
output ships as a new file. Loaded with importlib.resources, so the
packaged copy is used outside a git checkout too (same approach as the
JSON schemas).
"""
