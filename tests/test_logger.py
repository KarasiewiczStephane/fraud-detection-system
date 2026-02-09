"""Tests for src/utils/logger.py."""

import json
import logging

import pytest

from src.utils.logger import JSONFormatter, get_logger


@pytest.fixture(autouse=True)
def _cleanup():
    """Remove test loggers after each test to avoid handler accumulation."""
    yield
    for name in ("test_logger", "test_level", "test_exc"):
        lg = logging.getLogger(name)
        lg.handlers.clear()


def test_json_formatter_output():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="hello world",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["message"] == "hello world"
    assert "timestamp" in parsed
    assert "module" in parsed


def test_json_formatter_includes_exception():
    formatter = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg="something broke",
        args=(),
        exc_info=exc_info,
    )
    output = formatter.format(record)
    parsed = json.loads(output)

    assert "exception" in parsed
    assert "ValueError" in parsed["exception"]


def test_get_logger_returns_logger():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO


def test_get_logger_respects_level():
    logger = get_logger("test_level", level="DEBUG")
    assert logger.level == logging.DEBUG


def test_get_logger_emits_json(capsys):
    logger = get_logger("test_exc", level="INFO")
    logger.info("test message")
    captured = capsys.readouterr()
    # Logger writes to stderr
    parsed = json.loads(captured.err.strip())
    assert parsed["message"] == "test message"
    assert parsed["level"] == "INFO"


def test_get_logger_no_duplicate_handlers():
    logger = get_logger("test_logger")
    count = len(logger.handlers)
    get_logger("test_logger")  # second call
    assert len(logger.handlers) == count
