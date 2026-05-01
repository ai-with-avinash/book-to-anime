"""Language-provider adapters.

Each adapter module self-registers with :mod:`booktoanime.providers.registry`
on import. The registry imports them lazily on first use; missing optional
SDKs are tolerated so the default install stays small.
"""
