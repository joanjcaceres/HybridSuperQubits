## [v0.9.3] - 2025-06-28
### Improvements
- Modernized CI/CD with updated GitHub Actions workflows
- Migrated from `flake8` + `black` to `ruff` for linting and formatting
- Updated all type hints to use modern Python built-in types (dict, list, tuple)
- Enhanced regression tests with physically realistic parameters
- Fixed 136+ linting errors automatically
- Improved `pyproject.toml` configuration with comprehensive tool settings

### Breaking Changes
- **Minimum Python version updated to 3.10** (due to QuTiP dependency requirements)

### Testing
- Updated all regression test reference values with realistic physical parameters
- Increased Ferbo test dimension to 200 for better accuracy
- All 102 tests passing with new configuration
- Added robust parameter validation and error handling

### Documentation
- Updated project guidelines for English-only requirements
- Improved code documentation and type hints
- Cleaned up development dependencies

### Technical Details
- Ferbo tests now use realistic parameters: El~0.1-0.2 GHz, Ec~1-5 GHz, Ej~1-10 GHz
- Fluxonium tests use dimension=100 with physically consistent parameters
- Simplified build process without manual SciPy pre-installation
- Enhanced mypy configuration for gradual typing support
- Updated Poetry lock file for dependency compatibility

## [v0.9.2] - 2025-06-25
### Improvements
- Modernized CI/CD with updated GitHub Actions workflows
- Migrated from `flake8` + `black` to `ruff` for linting and formatting
- Updated all type hints to use modern Python built-in types (dict, list, tuple)
- Enhanced regression tests with physically realistic parameters
- Fixed 136+ linting errors automatically
- Improved `pyproject.toml` configuration with comprehensive tool settings

### Testing
- Updated all regression test reference values with realistic physical parameters
- Increased Ferbo test dimension to 200 for better accuracy
- All 102 tests passing with new configuration
- Added robust parameter validation and error handling

### Documentation
- Updated project guidelines for English-only requirements
- Improved code documentation and type hints
- Cleaned up development dependencies

### Technical Details
- Ferbo tests now use realistic parameters: El~0.1-0.2 GHz, Ec~1-5 GHz, Ej~1-10 GHz
- Fluxonium tests use dimension=100 with physically consistent parameters
- Simplified build process without manual SciPy pre-installation
- Enhanced mypy configuration for gradual typing support

## [v0.9.1] - 2025-06-25
### Main changes
- Add show progress in get_matelements_vs_paramvals.

## [v0.9.0] - 2025-06-25
### Main changes
- Enhanced Gatemonium class to support multiple transmission channels
- Changed Delta parameter to have a default value of 44 (for Aluminum)
- Reorganized Gatemonium potential calculation to handle multiple channels
- Updated derivatives with respect to external phase to support multiple channels

## [v0.8.9] - 2025-06-25
### Main changes
- Added Resonator class to simulate microwave resonators
- Improved JJA class with enhanced error handling and parameter validation
- Enhanced Ferbo class with optimized calculations and additional utilities
- Added scqubits as a dependency for enhanced compatibility
- Fixed several bugs and improved code quality
- Updated documentation

## [v0.8.4] - 2025-06-04
### Main changes
- Add version retrieval in the `__init__.py`.

## [v0.2.0] - 2025-02-21
### Main changes
- Add methods  `state_to_density_matrix`, `ptrace` and `purity` in ``operators.py``.
- Replace the Ferbo potential by one rotated by pi/2 in the Y axis.
- Fix bugs for updating the SpectrumData.
- Migrate project configuration to Poetry.
- Add ``CHANGELOG.md``.
