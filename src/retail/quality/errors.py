"""Erreurs de données : volontairement distinctes des pannes techniques (retry Prefect)."""


class QualityError(RuntimeError):
    """Lot hors schéma ou hors seuils : pas de retry, pas d'entraînement."""


class QualityGateBypassError(RuntimeError):
    """Le lot invalide de démo a été accepté : la porte qualité ne bloque pas."""
