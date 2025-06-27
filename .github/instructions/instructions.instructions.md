---
applyTo: '**'
---
# HybridSuperQubits Project Guidelines

## General Coding Standards
- Always follow best practices for Python development.
- Use PEP 8 style guidelines for all Python code.
- Code should be well-documented with docstrings following NumPy or Google style.
- Include appropriate type hints whenever possible.
- Write code that is clean, modular, and testable.

## Language
- All code, comments, docstrings, and documentation must be written in English.
- Variable names, function names, and class names should be descriptive and in English.
- Avoid non-English terminology in the codebase except for established scientific terms.
- All commit messages, PR descriptions, issue reports, and project documentation must be in English.
- Any generated summaries, reports, or analysis documents should be written in English.
- Communication with users and contributors should be in English to maintain consistency.

## Version Control and Deployment
- Be mindful of GitHub Actions workflows for CI/CD in the repository.
- When updating version numbers, follow semantic versioning (MAJOR.MINOR.PATCH).
- Always update CHANGELOG.md when making significant changes.
- Remember that PyPI deployment occurs via GitHub Actions when tags are pushed.
- Avoid duplicate version numbers - check existing versions on PyPI before publishing.

## Git Workflow
- Always create a new feature branch for each significant change or feature.
- Branch naming convention: `feat/feature-name` for features, `fix/issue-name` for bugfixes.
- Keep commits small, focused, and with clear messages.
- Before merging to main, ensure all tests pass and code meets quality standards.
- Use pull requests for code review before merging to main when working in teams.
- Regularly sync feature branches with main to minimize merge conflicts.

## Scientific Computing
- This is a quantum physics simulation package, so prioritize numerical stability and accuracy.
- Use appropriate scientific libraries (NumPy, SciPy, QuTiP) for quantum calculations.
- Ensure eigenvalue calculations and operator manipulations follow quantum mechanics principles.
- Physical constants and units should be handled consistently across the codebase.

## Testing and Documentation
- New features should include appropriate tests.
- Update documentation when adding new functionality.
- Example notebooks should demonstrate usage patterns of new features.
- Performance considerations are important for quantum simulations.

## Workflow Preferences
- Allow time for manual code review between implementation steps.
- Present changes incrementally for review before adding extensive examples.
- Prioritize code quality and correctness over rapid implementation.
- When suggesting multiple changes, implement them in stages to allow for feedback.