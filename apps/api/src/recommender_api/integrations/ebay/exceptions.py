"""Errors raised by the eBay integration."""


class EbayError(Exception):
    """Base eBay integration error."""


class EbayConfigurationError(EbayError):
    """eBay credentials are missing."""


class EbayAuthenticationError(EbayError):
    """eBay rejected OAuth credentials or an access token."""


class EbayRequestError(EbayError):
    """eBay returned an unsuccessful response."""


class EbayRateLimitError(EbayRequestError):
    """eBay rate limit was reached after retries."""


class EbayValidationError(EbayError):
    """An eBay response failed validation."""
