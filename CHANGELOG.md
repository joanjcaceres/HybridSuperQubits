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
