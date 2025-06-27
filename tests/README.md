# HybridSuperQubits Test Suite

This directory contains comprehensive tests for the HybridSuperQubits package, organized following industry best practices.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and test configuration
├── unit/                          # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── test_fluxonium.py         # Fluxonium qubit tests
│   └── test_ferbo.py             # Ferbo qubit tests
├── integration/                   # Integration tests (slower, full workflows)
│   ├── __init__.py
│   ├── test_spectrum_calculations.py  # Parameter sweeps, matrix elements
│   └── test_regression.py        # Regression tests with reference values
└── benchmarks/                    # Performance benchmarks
    ├── __init__.py
    └── test_performance.py       # Speed and memory usage tests
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Fast execution** (< 1 second per test)
- **Isolated functionality** (single methods/functions)
- **Mock external dependencies** when needed
- **High code coverage** of individual components

Example test categories:
- Initialization and parameter validation
- Operator calculations (Hermiticity, shapes, eigenvalues)
- Hamiltonian properties (symmetries, real eigenvalues)
- Mathematical relationships (commutation relations)

### Integration Tests (`tests/integration/`)
- **End-to-end workflows** (parameter sweeps, spectrum calculations)
- **Multi-component interactions** (qubits + utilities)
- **Consistency checks** between different calculation methods
- **Regression testing** with known reference values

Example test categories:
- Parameter sweep functionality
- Matrix element calculations
- Cross-validation between flux groupings
- Numerical consistency checks

### Benchmark Tests (`tests/benchmarks/`)
- **Performance measurement** and monitoring
- **Scaling behavior** with system size
- **Memory usage** validation
- **Speed comparisons** between implementations

Example test categories:
- Hamiltonian calculation speed
- Eigenvalue computation scaling
- Parameter sweep performance
- Memory efficiency

## Running Tests

### Prerequisites
Install test dependencies:
```bash
pip install pytest pytest-cov pytest-benchmark pytest-mock
# or
poetry install --with dev
```

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/                 # Only unit tests
pytest tests/integration/          # Only integration tests
pytest tests/benchmarks/           # Only benchmarks

# Run specific test files
pytest tests/unit/test_fluxonium.py
pytest tests/integration/test_regression.py
```

### Coverage Reporting
```bash
# Generate coverage report
pytest --cov=HybridSuperQubits --cov-report=html

# View HTML coverage report
open htmlcov/index.html
```

### Benchmark Execution
```bash
# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-only

# Save benchmark results
pytest tests/benchmarks/ --benchmark-save=baseline

# Compare against baseline
pytest tests/benchmarks/ --benchmark-compare=baseline
```

### Advanced Options
```bash
# Parallel execution (requires pytest-xdist)
pytest -n auto

# Stop on first failure
pytest -x

# Verbose output
pytest -v

# Run only tests matching pattern
pytest -k "test_hamiltonian"

# Run only failed tests from last run
pytest --lf
```

## Test Fixtures and Configuration

### Shared Fixtures (`conftest.py`)
Common test fixtures available to all tests:
- `fluxonium_params`: Standard Fluxonium parameters
- `ferbo_params`: Standard Ferbo parameters
- `fluxonium`: Initialized Fluxonium instance
- `ferbo`: Initialized Ferbo instance
- `tolerance`: Standard numerical tolerance (1e-10)
- `loose_tolerance`: Relaxed tolerance for noisy calculations (1e-6)

### Custom Markers
Define custom markers in `pytest.ini_options`:
```python
@pytest.mark.slow          # Long-running tests
@pytest.mark.numerical     # Tests requiring high precision
@pytest.mark.regression    # Regression tests with reference values
```

## Best Practices

### Writing Tests
1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Test both happy path and edge cases**
4. **Use appropriate fixtures** to reduce code duplication
5. **Include docstrings** for complex test logic
6. **Assert meaningful properties** (Hermiticity, normalization, etc.)

### Numerical Testing
1. **Use appropriate tolerances** based on calculation precision
2. **Test mathematical properties** (symmetries, conservation laws)
3. **Validate shapes and types** before numerical comparisons
4. **Check for NaN/Inf values** in results
5. **Test boundary conditions** and special cases

### Performance Testing
1. **Establish baseline measurements** for comparison
2. **Test scaling behavior** with different system sizes
3. **Monitor memory usage** for large calculations
4. **Compare alternative implementations** when available

### Regression Testing
1. **Store reference values** from validated implementations
2. **Update references** when intentional changes are made
3. **Use version control** to track reference value changes
4. **Document** the source of reference values

## Continuous Integration

Tests are automatically run on:
- Pull requests to main branch
- Pushes to main branch
- Nightly builds for performance monitoring

CI configuration includes:
- Multiple Python versions (3.9, 3.10, 3.11, 3.12)
- Different operating systems (Linux, macOS, Windows)
- Coverage reporting and badge updates
- Performance regression detection

## Contributing

When adding new features:
1. **Write tests first** (TDD approach when possible)
2. **Ensure good coverage** of new functionality
3. **Add regression tests** for critical calculations
4. **Update benchmarks** if performance-critical
5. **Document test additions** in PR description

When fixing bugs:
1. **Add test that reproduces the bug** first
2. **Verify test fails** before implementing fix
3. **Confirm test passes** after fix
4. **Consider edge cases** that might have similar issues

## Troubleshooting

### Common Issues
- **Import errors**: Ensure package is installed in development mode (`pip install -e .`)
- **Fixture not found**: Check `conftest.py` for fixture definitions
- **Slow tests**: Use `pytest --durations=10` to identify bottlenecks
- **Memory errors**: Run tests with smaller dimensions or fewer iterations

### Debug Mode
```bash
# Run with Python debugger
pytest --pdb

# Drop into debugger on first failure
pytest --pdb -x

# Capture stdout/stderr
pytest -s
```
