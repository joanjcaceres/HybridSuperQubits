# Contributing to HybridSuperQubits

Thank you for your interest in contributing to HybridSuperQubits! We welcome contributions from everyone and appreciate your effort to improve this project. Please follow the guidelines below to ensure a smooth and efficient collaboration.

---

## 📌 Contribution Guidelines

### 1. Choose Your Workflow

Depending on your level of access:

- **For Collaborators (with write access):**  
  Create a new branch directly in the main repository.

- **For External Contributors (without write access):**  
  Fork the repository on GitHub, clone your fork locally, and create a new branch. Then, submit a pull request from your branch to the main repository.

### 2. Setting Up Your Development Environment

If you plan to modify or extend the code, we recommend setting up a dedicated development environment.

1. **(Optional) Create a Virtual Environment**

   **Using Conda:**
   ```bash
   conda create --name hsq_dev python=3.10
   conda activate hsq_dev
   ```

   **Using venv:**
   ```bash
   python3 -m venv hsq_dev
   source hsq_dev/bin/activate  # macOS/Linux
   hsq_dev/Scripts/activate     # Windows
   ```

2. **Install in Editable Mode:**
   ```bash
   pip install -e .
   ```
   This lets you test your changes locally without needing to reinstall the package after every update.

### 3. Branching and Forking

- **For Collaborators:**  
  Create a new branch in the main repository using a descriptive name:
  ```bash
  git checkout -b feature/your-feature-name
  ```

- **For External Contributors:**  
  First, fork the repository. Then, clone your fork and create a branch:
  ```bash
  git clone https://github.com/your-username/HybridSuperQubits.git
  cd HybridSuperQubits
  git checkout -b feature/your-feature-name
  ```

### 4. Commit Messages and Branch Naming Conventions

- Use descriptive branch names (e.g., `feature/add-new-method`, `bugfix/fix-calculation-error`).
- Write clear, concise commit messages that explain the *what* and *why* of your changes.
- Follow the [Conventional Commits](https://www.conventionalcommits.org/) style if possible (e.g., `feat: add new plotting function` or `fix: correct energy calculation bug`).

### 5. Code Style and Documentation

- Follow **PEP 8** guidelines for Python code.
- Write descriptive names for functions, classes, and variables.
- Document your code using docstrings in the NumPy documentation style.
- Keep your changes modular and well-tested.

### 6. Testing

- Before submitting your pull request, run the tests to ensure everything works as expected:
  ```bash
  pytest tests/
  ```
- If you add a new feature or fix a bug, please include appropriate tests to cover your changes.

### 7. Submitting a Pull Request (PR)

- **For Collaborators:**  
  Push your branch directly to the repository:
  ```bash
  git push origin feature/your-feature-name
  ```

- **For External Contributors:**  
  Push your branch to your fork:
  ```bash
  git push origin feature/your-feature-name
  ```
  Then, open a pull request (PR) from your branch to the main branch of this repository.

- Include a clear title and detailed description in your PR, explaining your changes and the rationale behind them.
- Address any feedback promptly during the review process.

### 8. Code Review and Merging

- Your pull request will be reviewed by maintainers.
- Be prepared to discuss your implementation and make any necessary adjustments.
- Once approved, your changes will be merged into the main branch. We may squash your commits to keep the commit history clean.

---

## 💡 Contribution Areas

We welcome contributions in the following areas:
- 🛠 **New Features**: Adding new functionality.
- 🐛 **Bug Fixes**: Identifying and correcting issues.
- 📖 **Documentation**: Enhancing code comments, README, and other docs.
- ✅ **Testing**: Writing new tests or improving existing ones.
- 🚀 **Performance Improvements**: Optimizing the code for better performance.

---

## 📞 Contact and Support

If you have any questions or need assistance, please:
- Open an issue on GitHub.
- Use [GitHub Discussions](https://github.com/joanjcaceres/HybridSuperQubits/discussions) for broader topics.
- Reach out to one of the maintainers if necessary.

Happy coding and thank you for helping improve HybridSuperQubits! 🚀