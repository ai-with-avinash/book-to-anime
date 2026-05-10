"""Lip-sync providers.

Adapters in this subpackage produce a per-shot mp4 with mouth motion roughly
synced to the supplied narration WAV. They self-register via
:func:`booktoanime.providers.registry.register_lipsync_provider` so the
orchestrator can instantiate them from config.
"""
