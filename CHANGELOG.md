# Changelog

All notable changes to this project will be documented in this file.

## 11-09-2026

**commit:** `feat: removed mutual_info_classif from feature selection`

### Changed

- Removed mutual_info from feature selection
- Altered the all_feature_fs code.

## 11-09-2026

**commit:** `feat : updating evaluation code`

### Changed

- Altered the old evaluation code as per requirement.

## 11-09-2026

**commit:** `feat: add evaluation file`

### Changed

- Altered the code in evaluation file

## 11-09-2026

**commit:** `feat: add evaluation.py file in root folder`

### Added 

- Moved evaluation file from src to root.

## 11-09-2026

**commit:** `docs: Added the mermaid diagram and altered the docs`

### Added

- Added mermaid diagram to readme file

### Changed

- updated the docs file as per the code

## 10-09-2026

**commit:** `feat: add run_experiments to run entire project`

### Added

- Added run_experiments.py to run the complete experiment pipeline.
- Added support for running the project experiments from a single script.

### Changed

- Updated the project structure to support running the full experiment grid.
- Removed old generated result files and outdated presentation notebook.

## 10-09-2026

**commit:** `docs: update project documentation`

### Changed

- Updated the docs and altered some content

## 10-09-2026

**commit:** `feat: add result graphs and two  models`

### Added

- Added `results/graphs` the graphs of selected feturse along with score and in 'model.py' add to models.

## 10-09-2026

**commit:** `feat : add results of ml models in results folder`

### Added

- Added results of ml models into the result folder

## 07-09-2026

**commit:** `feat: add main.py file in root folder`

### Added

- Added `main.py` as the application entry point.


## 07-09-2026

**commit:** `docs: Added content in CHANGELOG and altered docs file`

### Added

- Added contents in changelog

### Changed

- altered some content in design and proposal files


## 06-09-2026

**commit:** `feat : add complete working ML pipeline`

### Changed

- changed some parts of proposal.md file


## 06-09-2026

**commit:** `feat : add complete working ML pipeline`

### Added

- Added implementation of data loading
- Added evaluation implementation
- Added feature_selection implementation
- Added model implementation
- Added application entry point main.py


## 06-09-2026

**commit:** `docs: update project documentation and file structure`

### Added

- Added architecture/proposal.md containing technical and architecture content of our project
- Added architecture/design.md containing technical and architectural details and contents of our project
- Added results/expreiment_results.csv for showing results

### Changed

- Renamed file name from src/feture_selection.py to src/feature_selection.py


## 06-09-2026

**commit:** `Initial commit`

### Added

- Added the initial project structure for the Human Activity Recognition project.
- Added architecture/ directory which consists design and proposal files.
- Added dataset/ directory for training and testing datasets.
- Added src/ directory for data loading, feature selection, model.
- Added results/ directory for experiment results and reporting notebook.
- Added run_experiment.py as the main experiment execution script.
- Added project configuration and documentation files including .gitignore, README.md, HANDOFF.md, CHANGELOG.md, and requirements.txt.
- Added basic loading functionality