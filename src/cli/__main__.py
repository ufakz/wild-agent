"""CLI entry point for wild-agent."""

import sys
from pathlib import Path

import structlog
import logging
from dotenv import load_dotenv

from src.cli.parser import cli


def setup_logging():
    """Configure structured logging for the application."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def main():
    """Main entry point for wild-agent CLI."""
    import logging
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Setup logging
    setup_logging()
    
    # Run CLI
    try:
        cli()
    except KeyboardInterrupt:
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
