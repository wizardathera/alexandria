"""
Logging configuration for the Alexandria application.

This module provides structured logging setup using Python's logging module
with custom formatting for better debugging and monitoring.
"""

import logging
import logging.config
import sys
from typing import Dict, Any
from pathlib import Path

from src.utils.config import get_settings


def setup_logger(name: str = None) -> logging.Logger:
    """
    Set up and configure logger for the application.
    
    Args:
        name (str, optional): Logger name. Defaults to root logger.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    settings = get_settings()
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Logging configuration
    logging_config: Dict[str, Any] = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': settings.log_level.upper(),
                'formatter': 'detailed' if settings.debug else 'simple',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': log_dir / 'alexandria.log',
                'mode': 'a',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            '': {  # Root logger
                'level': settings.log_level.upper(),
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Get logger instance
    logger = logging.getLogger(name)
    
    # Log startup information
    if name is None:  # Only log this once for root logger
        logger.info(f"Logger initialized - Environment: {settings.environment}")
        logger.info(f"Log level: {settings.log_level.upper()}")
        logger.debug("Debug logging enabled")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name (str): Logger name (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)