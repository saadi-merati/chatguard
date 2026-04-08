"""
utils/logger.py — Configuration du logging centralisé pour ChatGuard
"""

import logging
import sys
import os
from pathlib import Path


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "chatguard.log")

# Créer le répertoire de logs si nécessaire
Path("logs").mkdir(exist_ok=True)

# Format des messages
FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger configuré pour le module `name`.

    - Affiche les logs INFO+ dans la console (stdout)
    - Écrit tous les logs DEBUG+ dans logs/chatguard.log
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Déjà configuré

    logger.setLevel(logging.DEBUG)

    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(FORMATTER)
    logger.addHandler(console_handler)

    # Handler fichier
    try:
        file_handler = logging.FileHandler(f"logs/{LOG_FILE}", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FORMATTER)
        logger.addHandler(file_handler)
    except Exception:
        pass  # Ne pas bloquer si les logs fichier ne sont pas disponibles

    logger.propagate = False
    return logger
