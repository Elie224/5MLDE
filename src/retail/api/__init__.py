"""API FastAPI. ``create_app`` est importé à la demande pour éviter un import circulaire."""


def __getattr__(name: str):
    if name == "create_app":
        from retail.api.app import create_app

        return create_app
    raise AttributeError(name)


__all__ = ["create_app"]
