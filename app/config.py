import os

from dynaconf import Dynaconf

current_directory = os.path.dirname(os.path.realpath(__file__))

config = Dynaconf(
    envvar_prefix=False,
    env_switcher="ENV",
    environments=True,
    load_dotenv=True,
    settings_files=[f"{current_directory}/settings.yaml"],
    secrets=f"{current_directory}/secrets.yaml",
)
