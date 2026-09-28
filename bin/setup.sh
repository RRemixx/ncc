#!/usr/bin/env bash
# Run from bin/: installs the ncc package (editable, with the test extra) and creates the output directories.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Installing the ncc package..."
pip install -e "..[test]"

echo "Making directories..."
mkdir -p ../results/base_pred ../results/csvs ../models

echo "Done!"
