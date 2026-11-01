"""Package-specific fail-fast errors."""


class ContractError(ValueError):
    """Raised when an nnU-Net plan, tensor, or runtime violates its contract."""
