import getpass
import os

from dotenv import load_dotenv


def get_dev_name() -> str:
    """
    Retrieves the developer name from the environment variable DEV_NAME.
    If not set, falls back to the system username with a humorous prefix.
    """
    # Ensure env vars are loaded
    load_dotenv()

    dev_name = os.getenv("DEV_NAME")
    if dev_name:
        return dev_name

    # Fallback
    try:
        username = getpass.getuser()
    except Exception:
        # Fallback for some systems where getpass might fail
        username = os.environ.get("USERNAME", "unknown-user")

    return f"I-didnt-RTFM-{username}"
