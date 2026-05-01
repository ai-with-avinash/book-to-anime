# Providers

BookToAnime defines three abstract provider interfaces:
:class:`LanguageProvider`, :class:`AudioProvider`, :class:`VisualProvider`.
The pipeline never imports concrete adapters — they're discovered through
the registry.

## Bring your own provider

1. Create a module under
   `src/booktoanime/providers/{language,audio,visual}/myprovider.py`.

2. Implement the abstract interface from `booktoanime.providers.base`.

3. Register the factory:

   ```python
   from booktoanime.providers import register_language_provider

   @register_language_provider("myprovider")
   def _factory(sub_config):
       return MyProvider(...)
   ```

4. Add the module to the `_BUILTIN_*_MODULES` tuple in
   `booktoanime/providers/registry.py` (or import it explicitly from your
   own startup code if you don't want to fork).

5. Configure it from `config.yaml`:

   ```yaml
   language:
     active: myprovider
     myprovider:
       api_key_env: MYPROVIDER_API_KEY
       model: my-model
   ```

## Contracts the orchestrator depends on

### Async-only

Every method on every provider is `async`. Implementations that wrap a
synchronous SDK MUST hop calls to `asyncio.to_thread`. Don't block the
event loop — the SSE bus, the API, and other shots all run on the same
loop.

### Typed errors

Provider failures should be mapped onto the `booktoanime.errors` hierarchy:

| Upstream signal | Adapter raises | Orchestrator behavior |
|---|---|---|
| 401 / 403 / bad key | `ProviderAuthError` | No retry; stage fails immediately |
| 429 / quota | `ProviderRateLimitError` | Retried with jittered backoff |
| Network / 5xx / timeout | `ProviderTransientError` | Retried with jittered backoff |
| Vision call against a text-only model | `CapabilityNotSupportedError` | Tries `language.vision_fallback`, else degrades |
| Anything else | `ProviderError` | Stage fails; user gets the `user_message` |

### Cancellation

Long-running calls MUST propagate `asyncio.CancelledError`. Tenacity is
configured to re-raise `CancelledError` correctly; if you write your own
retry loop, do the same.

### Resource ownership

`close()` is called once per job. Adapters that build their own HTTP client
should `close()` it; adapters that accept an injected client (used heavily
in tests) should track ownership and only close clients they built.

### Determinism

Adapters that accept `seed:` should pass it through. The image renderer
relies on stable seeds for resume to be a no-op when re-running with the
same shot list.

## Vision capability

The orchestrator passes embedded PDF images to `LanguageProvider.explain_image`.
If your provider can't do vision, **raise `CapabilityNotSupportedError`** —
don't return an empty `ImageExplanation`. The orchestrator then routes to
`language.vision_fallback` (a separately-configured provider name) or, if
none is set, falls back to a text-only synthesis from the PDF caption +
surrounding context.
