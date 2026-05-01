"""Visual (image-generation) provider adapters.

Each adapter module self-registers with :mod:`booktoanime.providers.registry`
on import. Heavy dependencies (torch, diffusers) are install extras so the
default install stays small.
"""
