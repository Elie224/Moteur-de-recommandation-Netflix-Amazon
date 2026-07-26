"""Errors raised by the TMDB integration."""


class TMDBError(Exception):
    """Base error for TMDB integration."""


class TMDBConfigurationError(TMDBError):
    """Raised when TMDB credentials are missing."""


class TMDBRequestError(TMDBError):
    """Raised when TMDB returns an invalid response."""


class TMDBRateLimitError(TMDBRequestError):
    """Raised when TMDB returns HTTP 429 after retries."""
