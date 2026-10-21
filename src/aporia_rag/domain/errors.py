"""The application's exceptions.

Adapters translate library-specific failures (`ollama.ResponseError`,
`httpx.ConnectError`, Chroma's errors...) into these, so code outside
`adapters/` never imports a provider's exception types, and the CLI can
report any expected failure with one `except AporiaError` (plus `OSError`,
the environment's own: a missing file, a full disk).

    AporiaError
    ├── ConfigurationError          settings that cannot work (unknown provider...)
    ├── CorpusError                 a corpus text is unavailable or not the expected edition
    ├── ProviderError               a model or web search provider failed
    │   ├── ProviderUnavailableError    ...because it cannot be reached
    │   └── ModelNotAvailableError      ...because the model is not installed/allowed
    ├── IndexMismatchError          an index built in a different embedding space
    └── VectorDimensionError        vectors of the wrong size (also a ValueError)
"""


class AporiaError(Exception):
    """Base class for every error the application expects and reports."""


class ConfigurationError(AporiaError):
    """The settings name something that does not exist or cannot work together."""


class CorpusError(AporiaError):
    """A corpus text cannot be downloaded, or is not the edition its manifest entry describes."""


class ProviderError(AporiaError):
    """A provider (embedding, chat or web search) failed or answered nonsense."""


class ProviderUnavailableError(ProviderError):
    """The provider cannot be reached: not running, no network, wrong host."""


class ModelNotAvailableError(ProviderError):
    """The provider is reachable but does not have the requested model."""


class IndexMismatchError(AporiaError):
    """Vectors from one embedding space were about to meet another.

    Similarities between vectors from different models, sizes or prompt
    formats are meaningless, so this is always refused rather than warned.
    """


class VectorDimensionError(AporiaError, ValueError):
    """A vector does not have the size of the others it is stored or compared with."""
