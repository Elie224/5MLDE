class ReplayError(RuntimeError):
    """Rejeu hebdomadaire impossible."""


class ReplayDone(ReplayError):
    """Tous les lots du manifeste ont déjà été livrés."""
