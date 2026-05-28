# Test Suite for GIPEX

This directory contains comprehensive pytest test suites for the GIPEX (Geospatial Indicators for Proxy Environmental eXposure) application.

## Test Files

### `test_extraction.py`
Comprehensive tests for the `extraction.py` module, covering:

- **GEE Initialization**: Tests for Google Earth Engine authentication and initialization
- **Scale/Offset Helpers**: Tests for data scaling and offset functions
- **Extraction Functions**: Tests for core data extraction from various satellite sources
- **Specific Extractors**: Tests for ERA5, Sentinel-2, MODIS, VIIRS, and other data sources
- **Variable Groups**: Tests for variable grouping and getter building
- **Post-Processing**: Tests for derived meteorology, gap filling, and road metrics
- **ExtractionRunner**: Tests for the background extraction runner class
- **Cloud Masking**: Tests for Sentinel-2 cloud masking
- **Index Calculations**: Tests for spectral index calculations (NDVI, NDBI, etc.)
- **Integration Tests**: End-to-end workflow tests
- **Edge Cases**: Tests for error handling and edge cases

**Test Coverage**: 717 lines, 80+ test cases

### `test_app.py`
Comprehensive tests for the `app.py` Dash application, covering:

- **App Constants**: Tests for application configuration and constants
- **Module Availability**: Tests for checking optional module availability
- **Roads Database**: Tests for OSM roads database functions
- **Data Helpers**: Tests for data processing helper functions
- **File Parsing**: Tests for CSV, Excel, JSON, Parquet, and NetCDF parsing
- **Statistics**: Tests for parallel statistics computation
- **Visualization**: Tests for map and chart generation
- **Data Type Conversions**: Tests for type coercion and date parsing
- **Integration Tests**: End-to-end data pipeline tests
- **Configuration**: Tests for configuration validation
- **Performance**: Tests for performance-critical operations
- **Error Handling**: Tests for error messages and graceful failures

**Test Coverage**: 783 lines, 90+ test cases

## Running Tests

### Install Test Dependencies

```bash
# Install the package with test dependencies
pip install -e ".[test]"
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# Run tests in parallel (faster)
pytest tests/ -n auto
```

### Run Specific Test Files

```bash
# Run only extraction tests
pytest tests/test_extraction.py -v

# Run only app tests
pytest tests/test_app.py -v
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
pytest tests/test_extraction.py::TestGEEInitialization -v

# Run a specific test function
pytest tests/test_app.py::TestDataHelpers::test_get_special_cols_standard -v

# Run tests matching a pattern
pytest tests/ -k "test_parse" -v
```

### Run with Different Output Formats

```bash
# Short traceback
pytest tests/ --tb=short

# No traceback
pytest tests/ --tb=no

# Show local variables in traceback
pytest tests/ -l

# Stop on first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s
```

## Test Structure

Each test file follows this structure:

1. **Imports**: All necessary imports including the module under test
2. **Fixtures**: Reusable test data and mock objects
3. **Test Classes**: Organized by functionality
4. **Test Methods**: Individual test cases with descriptive names

## Mocking Strategy

The tests use extensive mocking to:
- Avoid actual GEE API calls (expensive and requires authentication)
- Simulate file I/O operations
- Test error conditions
- Ensure tests run quickly and reliably

Key mocked components:
- Google Earth Engine API (`ee` module)
- File system operations
- Database connections
- Network requests

## Test Data

Test fixtures provide:
- Sample DataFrames with realistic data
- Mock CSV/Excel content
- Simulated GEE responses
- Temporary directories for file operations

## Coverage Goals

Target coverage areas:
- ✅ Core extraction functions
- ✅ Data processing pipelines
- ✅ File parsing and validation
- ✅ Visualization generation
- ✅ Error handling
- ✅ Edge cases (empty data, invalid inputs, etc.)

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- No external dependencies required (all mocked)
- Fast execution (< 30 seconds for full suite)
- Deterministic results
- Clear failure messages

## Adding New Tests

When adding new functionality:

1. Add corresponding test cases
2. Follow existing naming conventions
3. Use appropriate fixtures
4. Mock external dependencies
5. Test both success and failure paths
6. Include edge cases

Example test structure:

```python
class TestNewFeature:
    """Test new feature functionality"""

    def test_basic_functionality(self, sample_dataframe):
        """Test basic use case"""
        result = new_feature(sample_dataframe)
        assert result is not None

    def test_edge_case(self):
        """Test edge case handling"""
        result = new_feature(pd.DataFrame())
        assert result == expected_value

    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            new_feature(invalid_input)
```

## Known Limitations

- GEE API calls are mocked (no actual satellite data retrieval)
- Some tests require optional dependencies (xarray, geopandas)
- Road metrics tests require shapefile data
- Performance tests may vary by system

## Troubleshooting

### Import Errors
If you see import errors, ensure all dependencies are installed:
```bash
pip install -e ".[test]"
```

### Skipped Tests
Some tests are skipped if optional dependencies are missing:
```bash
# Install optional dependencies
pip install xarray netCDF4 geopandas
```

### Slow Tests
Use parallel execution:
```bash
pytest tests/ -n auto
```

## Contributing

When contributing tests:
1. Ensure all tests pass locally
2. Add docstrings to test functions
3. Use descriptive test names
4. Group related tests in classes
5. Update this README if adding new test categories

## Contact

For questions about the test suite:
- Check existing test examples
- Review pytest documentation: https://docs.pytest.org/
- Consult the main project README
