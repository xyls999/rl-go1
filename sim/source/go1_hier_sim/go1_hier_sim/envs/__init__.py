"""Real Isaac Lab environments; import the simulator only after AppLauncher."""


def make_env(**kwargs):
    from .environment import make_env as create

    return create(**kwargs)
