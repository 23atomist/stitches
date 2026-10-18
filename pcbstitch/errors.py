"""Exception types raised by pcbstitch."""


class StitchError(Exception):
    """Base class for all pcbstitch errors."""


class ScanError(StitchError):
    """The scan directory is missing, empty, or malformed."""


class RegistrationError(StitchError):
    """Tiles could not be aligned."""


class StaticDetectionError(StitchError):
    """Static-region detection could not find a usable crop."""


class ConfigError(StitchError):
    """A configuration value is invalid or implausible for this scan."""
