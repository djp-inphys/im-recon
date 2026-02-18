# Logging Implementation

This document describes the logging functionality that has been added to the model processing system.

## Overview

Comprehensive logging has been added to all major modules to improve debugging, monitoring, and operational visibility.

## Configuration

The logging system is configured in `data.py` with the following features:

- **Log Level**: INFO (can be changed to DEBUG for more detailed output)
- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Handlers**:
  - File handler: Writes to `model_processing.log`
  - Console handler: Displays logs in the terminal

## Log Levels Used

- **INFO**: General operational information (initialization, progress, completion)
- **DEBUG**: Detailed diagnostic information (data shapes, processing steps)
- **WARNING**: Non-critical issues that should be noted
- **ERROR**: Error conditions with full exception details

## Modules with Logging

### 1. data.py
- Initialization of obsData class
- File loading operations (gamma index, gamma observations)
- Data processing operations
- Phase count management
- File I/O operations

### 2. model.py
- Model initialization
- E2ADC lookup table processing
- Channel mapping setup
- Model data loading

### 3. main.py
- Application startup/shutdown
- Processing loop progress (every 100 instances)
- Phase transitions
- Overall application flow

### 4. splines.py
- Spline interpolation operations
- Image reshaping and processing

## Log File

The log file `model_processing.log` contains:
- Timestamped entries for all operations
- Progress tracking for long-running processes
- Error details with stack traces
- Configuration and initialization information

## Usage Examples

### Viewing Logs
```bash
# View the log file
cat model_processing.log

# Follow logs in real-time
tail -f model_processing.log

# View recent entries
tail -n 50 model_processing.log
```

### Changing Log Level
To enable more detailed DEBUG logging, modify the logging configuration in `data.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_processing.log'),
        logging.StreamHandler()
    ]
)
```

## Benefits

1. **Debugging**: Detailed information about data processing steps
2. **Monitoring**: Progress tracking for long-running operations
3. **Error Diagnosis**: Full exception details with context
4. **Performance Analysis**: Timing information for major operations
5. **Operational Visibility**: Clear understanding of what the system is doing

## Testing

Run `test_logging.py` to verify that logging is working correctly:

```bash
python test_logging.py
```

This will test:
- Basic logging functionality
- Module imports with logging
- Log file creation
- Recent log entries display

## Log Rotation

For production use, consider implementing log rotation to prevent log files from growing too large:

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'model_processing.log', 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```
