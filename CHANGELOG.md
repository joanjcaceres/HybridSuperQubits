## [Unreleased]

## [v0.12.0] - 2026-05-16
### New Features
- Extended `HybridSuperQubits.noise` with table-level sweep functions:
  `t1_table_from_spectral_density`, `tphi_1_over_f_table`, `tphi_cqps_table`.
  Channel-specific spectral densities (`S_capacitive`, `S_inductive`,
  `S_flux_bias_line`, `S_charge_impedance`, `S_one_over_f_flux`,
  `S_critical_current`, `S_andreev`) are now public — a single source of truth
  shared by both the single-shot and sweep code paths.
- The 7 `get_t1_*_vs_paramvals` methods, 2 `get_tphi_*_vs_paramvals` methods,
  `get_tphi_CQPS_vs_paramvals`, and the internal `_get_t1_vs_paramvals` /
  `_get_tphi_1_over_f_vs_paramvals` dispatchers on `QubitBase` are now thin
  wrappers around these functions. Phase B of the #17 refactor.

### Breaking Changes
- **Reconciled single-shot vs sweep physics divergences.** The single-shot
  `t1_capacitive`, `t1_inductive`, and `t1_flux_bias_line` methods (and the
  corresponding `HybridSuperQubits.noise` functions) now produce the same
  numerical results as their sweep counterparts. Previously they differed by
  factors of ~2 (capacitive) and used a different formula entirely
  (flux-bias-line). The sweep version was the physically correct one and is
  now the single source of truth.
- Removed the `total: bool` keyword argument from `t1_capacitive`,
  `t1_inductive`, `t1_flux_bias_line` (both class methods and standalone
  functions) and from `get_t1_*_vs_paramvals` sweep methods. The kwarg was
  the knob that selected the buggy "SD(+ω) + SD(-ω)" summing in single-shot;
  with the physics reconciled there is no longer a switch to flip. Passing
  `total=...` will raise `TypeError`.

### Bug Fixes
- Capacitive single-shot T1 prefactor corrected from 8 to 16 to match the
  sweep path. Combined with the v0.11.0 `Q_cap` default fix, single-shot
  capacitive T1 values are now consistent with sweep values for the same
  parameters.

## [v0.11.0] - 2026-05-16
### New Features
- Added `HybridSuperQubits.noise` module: standalone pure functions for the existing
  decoherence formulas (`t1_capacitive`, `t1_inductive`, `t1_flux_bias_line`,
  `tphi_1_over_f`, `tphi_CQPS`, plus the generic `t1_from_spectral_density`
  builder). Functions take eigenvalues, operator matrix elements, and scalar
  parameters — no qubit object required — so decoherence can be computed for
  any user-supplied Hamiltonian. First step of the functional refactor proposed
  in #17.
- The corresponding methods on `QubitBase` are now thin wrappers that delegate
  to `HybridSuperQubits.noise` (covered by `tests/integration/test_noise_parity.py`).

### Bug Fixes
- Corrected the default capacitive quality factor prefactor in `t1_capacitive`
  from `1e6` to `1/3e-5` (~3.33e4), the physically correct value already in
  use by the `get_t1_capacitive_vs_paramvals` sweep. This makes the single-shot
  and sweep capacitive-T1 paths consistent. **Behavior change:** capacitive T1
  values returned by `qubit.t1_capacitive()` (when `Q_cap` is not explicitly
  supplied) are now ~30x shorter than in v0.10.x.

## [v0.10.9] - 2026-04-15
### Bug Fixes
- Added missing `4 * Ec` prefactor to the scalar Berry contribution in `Ferbo._berry_contribution()`.

## [v0.10.8] - 2026-04-11
### Breaking Changes
- Renamed Ferbo basis values from `"static"` → `"ballistic"` and `"adiabatic"` → `"Andreev"` across the codebase.
- Renamed the `andreev_basis` parameter to `representation` in `Ferbo.wavefunction()`. The parameter name remains `andreev_basis` in `Ferbo.plot_wavefunction()`.

## [v0.10.7] - 2026-04-04
### Bug Fixes
- Fixed CI benchmark workflow referencing removed `full` extra after dependencies were made required.

## [v0.10.6] - 2026-04-04
### Improvements
- Removed `poetry.lock` from version control; not needed for a library package and was causing CI build failures when dependencies changed.

## [v0.10.5] - 2026-04-04
### Improvements
- Made `numpy`, `scipy`, and `qutip` required dependencies so `pip install HybridSuperQubits` is self-contained. Removed unused `scqubits` from required deps (kept as optional extra).
- Removed `qutip.Qobj` dependency from `cos_phi`/`sin_phi` in `operators.py`; they now return plain scipy sparse matrices.

## [v0.10.4] - 2026-04-04
### Bug Fixes
- Fixed missing wavefunction normalization factor in `Gatemonium.wavefunction()`. Added the `1/sqrt(phase_zpf)` divisor to match the correct normalization used in `Ferbo` and `Fluxonium`.

## [v0.10.3] - 2026-04-04
### Improvements
- Replaced `str` type annotations with `Literal` types for all fixed-option string parameters across `Andreev`, `Ferbo`, `Fluxonium`, `Gatemonium`, `QubitBase`, and `Resonator` classes, improving IDE autocompletion and type safety.

## [v0.10.2] - 2026-04-03
### Breaking Changes
- Replaced the `rotate` boolean in `Ferbo.wavefunction()` (now `representation` parameter, `"ballistic"` | `"Andreev"`) and `Ferbo.plot_wavefunction()` (now `andreev_basis` parameter, `"ballistic"` | `"Andreev"`).

### New Features
- Added scalar Berry phase correction to the Andreev potential via `Ferbo._berry_contribution()`.
- Added `include_berry` flag to `Ferbo.potential()` to toggle the Berry correction.

### Testing
- Added unit tests for Berry contribution (include/exclude, both `ABS` and `EL` flux groupings).
- Added test verifying `return_evecs` and evals-only paths return consistent eigenvalues.

## [v0.10.1] - 2026-02-13
### Added
- Added second derivative methods for Ferbo class: `d2_hamiltonian_d_Gamma2`, `d2_hamiltonian_d_deltaGamma2`, and `d2_hamiltonian_d_er2`

## [v0.10.0] - 2026-02-13
### Improvements
- Fixing factors in the spectrum density value and refactoring

## [v0.9.4] - 2025-10-31
### Improvements
- Version bump for patch release

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
