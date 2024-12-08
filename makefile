# Makefile

# Variables
PYTHON = python
PIP = pip
VENV_DIR = venv
REQUIREMENTS = requirements.txt

# Virtual environment setup
$(VENV_DIR)/bin/activate: $(VENV_DIR)
	$(PYTHON) -m venv $(VENV_DIR)
    $(VENV_DIR)/bin/$(PIP) install -r $(REQUIREMENTS)

# Install dependencies
install: $(VENV_DIR)/bin/activate

# Run the main application
run: install
    $(VENV_DIR)/bin/$(PYTHON) gui/main.py

# Clean the virtual environment
clean:
    rm -rf $(VENV_DIR)

.PHONY: install run clean