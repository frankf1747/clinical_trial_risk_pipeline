class DataGateError(RuntimeError):
    """A data sanity check failed. The pipeline stops rather than produce a misleading result."""
