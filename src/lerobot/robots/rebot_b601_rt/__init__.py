from .config_rebot_b601_rt import RebotB601RTConfig

try:
    import rebot_rt  # noqa: F401
except ImportError:
    pass
else:
    from .rebot_b601_rt import RebotB601RT
