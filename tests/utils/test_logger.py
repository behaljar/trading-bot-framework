"""
Tests for logger utility.
"""

import pytest
import logging
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from framework.utils.logger import setup_logger, JSONFormatter


class TestJSONFormatter:
    """Test cases for JSONFormatter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = JSONFormatter()

    def test_format_basic_record(self):
        """Test formatting of basic log record."""
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='test.py',
            lineno=42,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = self.formatter.format(record)
        
        # Should be valid JSON
        log_data = json.loads(formatted)
        
        # Check required fields
        assert log_data['timestamp']
        assert log_data['level'] == 'INFO'
        assert log_data['message'] == 'Test message'
        assert log_data['module'] == 'test'
        assert log_data['line'] == 42

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        record = logging.LogRecord(
            name='trading_logger',
            level=logging.WARNING,
            pathname='strategy.py',
            lineno=100,
            msg='Signal generated',
            args=(),
            exc_info=None
        )
        
        # Add extra attributes
        record.symbol = 'BTC_USDT'
        record.signal_type = 'buy'
        record.price = 50000.0
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        # Check extra fields are included
        assert log_data['symbol'] == 'BTC_USDT'
        assert log_data['signal_type'] == 'buy'
        assert log_data['price'] == 50000.0

    def test_format_with_exception(self):
        """Test formatting with exception information."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name='test_logger',
                level=logging.ERROR,
                pathname='test.py',
                lineno=50,
                msg='Error occurred',
                args=(),
                exc_info=True
            )
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        # Should include exception info
        assert 'exception' in log_data
        assert 'ValueError' in log_data['exception']
        assert 'Test exception' in log_data['exception']

    def test_format_excludes_internal_fields(self):
        """Test that internal logging fields are excluded."""
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='test.py',
            lineno=42,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        # These fields should not be in the output
        excluded_fields = [
            'name', 'levelno', 'pathname', 'filename', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
            'processName', 'process', 'args', 'exc_info', 'exc_text', 'stack_info'
        ]
        
        for field in excluded_fields:
            assert field not in log_data


class TestSetupLogger:
    """Test cases for setup_logger function."""

    def test_setup_logger_default(self):
        """Test logger setup with default parameters."""
        logger = setup_logger()
        
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        assert logger.name == 'trading_bot'

    def test_setup_logger_custom_level(self):
        """Test logger setup with custom log level."""
        logger = setup_logger(level="DEBUG")
        
        assert logger.level == logging.DEBUG

    def test_setup_logger_custom_name(self):
        """Test logger setup with custom logger name."""
        logger = setup_logger(name="custom_logger")
        
        assert logger.name == "custom_logger"

    def test_setup_logger_invalid_level(self):
        """Test logger setup with invalid log level."""
        # Should handle gracefully and use default
        logger = setup_logger(level="INVALID")
        
        # Should still create a logger (might use default level)
        assert isinstance(logger, logging.Logger)

    @patch('framework.utils.logger.Path')
    def test_setup_logger_creates_log_directory(self, mock_path):
        """Test that logger setup creates log directory."""
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        
        setup_logger()
        
        # Should call mkdir with parents=True, exist_ok=True
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_setup_logger_file_handler(self):
        """Test that file handler is configured correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the logs directory to be in temp dir
            with patch('framework.utils.logger.Path') as mock_path:
                log_dir = Path(temp_dir) / 'logs'
                log_dir.mkdir(exist_ok=True)
                mock_path.return_value = log_dir
                
                logger = setup_logger()
                
                # Should have handlers configured
                assert len(logger.handlers) > 0
                
                # Check that at least one handler uses JSONFormatter
                has_json_formatter = any(
                    isinstance(handler.formatter, JSONFormatter)
                    for handler in logger.handlers
                )
                assert has_json_formatter

    def test_setup_logger_console_handler(self):
        """Test that console handler is configured."""
        logger = setup_logger()
        
        # Should have handlers
        assert len(logger.handlers) > 0
        
        # Check handler types
        handler_types = [type(handler).__name__ for handler in logger.handlers]
        
        # Should have either StreamHandler or other console handler
        has_console_handler = any(
            'Stream' in handler_type or 'Console' in handler_type
            for handler_type in handler_types
        )
        assert has_console_handler

    def test_logger_functionality(self):
        """Test that logger actually logs messages."""
        logger = setup_logger(level="DEBUG")
        
        # Test that we can log without errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # If we got here without exceptions, logging is working

    def test_logger_with_extra_fields(self):
        """Test logging with extra fields."""
        logger = setup_logger()
        
        # Test logging with extra context
        logger.info("Trade executed", extra={
            'symbol': 'BTC_USDT',
            'price': 50000.0,
            'quantity': 0.1,
            'side': 'buy'
        })
        
        # Should not raise exceptions

    def test_multiple_logger_instances(self):
        """Test that multiple logger instances work correctly."""
        logger1 = setup_logger(name="logger1")
        logger2 = setup_logger(name="logger2")
        
        assert logger1.name != logger2.name
        assert logger1 is not logger2
        
        # Both should be functional
        logger1.info("Message from logger1")
        logger2.info("Message from logger2")