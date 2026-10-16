from .config_bi_rebot_b601_rt import BiRebotB601RTConfig

try:
    import rebot_rt  # noqa: F401
except ImportError:
    pass
else:
    from .bi_rebot_b601_rt import BiRebotB601RT
