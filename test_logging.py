#!/usr/bin/env python3
"""
Simple test script to verify logging functionality
"""

import logging
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_logging_setup():
    """Test that logging is properly configured"""
    
    # Test basic logging functionality
    logger = logging.getLogger('test_logger')
    logger.info("Testing basic logging functionality")
    
    # Test that log file is created
    log_file = 'model_processing.log'
    if os.path.exists(log_file):
        print(f"✓ Log file {log_file} exists")
        
        # Read and display recent log entries
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                print("✓ Log file has content")
                print("Recent log entries:")
                for line in lines[-5:]:  # Show last 5 lines
                    print(f"  {line.strip()}")
            else:
                print("⚠ Log file exists but is empty")
    else:
        print(f"⚠ Log file {log_file} not found")
    
    # Test importing modules with logging
    try:
        print("Testing module imports with logging...")
        
        # Test data module
        from data import logger as data_logger
        data_logger.info("Testing data module logging")
        print("✓ data module logging works")
        
        # Test model module  
        from model import logger as model_logger
        model_logger.info("Testing model module logging")
        print("✓ model module logging works")
        
        # Test main module
        from main import logger as main_logger
        main_logger.info("Testing main module logging")
        print("✓ main module logging works")
        
        # Test splines module
        from splines import logger as splines_logger
        splines_logger.info("Testing splines module logging")
        print("✓ splines module logging works")
        
        print("\n✓ All logging tests passed!")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
    except Exception as e:
        print(f"✗ Error testing logging: {e}")

if __name__ == "__main__":
    print("Testing logging configuration...")
    test_logging_setup()
