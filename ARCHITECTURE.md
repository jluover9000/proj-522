# Project Architecture & Diagrams

This document contains detailed architectural diagrams for the Bank Marketing Analysis project.

## Table of Contents
- [Project Structure](#project-structure)
- [Scripts vs Tests Workflow](#scripts-vs-tests-workflow)
- [Developer Workflow](#developer-workflow)

---

## Project Structure

This project follows a modular architecture separating concerns between reusable functions, CLI orchestration, and testing:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROJECT STRUCTURE                            │
│                                                                   │
│  src/                    scripts/                  tests/        │
│  ├── __init__.py         ├── 01_download_data.py  ├── conftest.py│
│  ├── download_data.py    ├── 02_clean_validate... ├── test_down...│
│  ├── preprocess.py       ├── 03_eda.py            ├── test_prep...│
│  ├── eda.py              ├── 04_fit_model.py      ├── test_eda.py│
│  ├── model_training.py   └── 05_evaluate_model.py ├── test_mode...│
│  └── model_evaluation.py                          └── test_mode...│
│                                                                   │
│  [Pure Functions]        [CLI Orchestration]      [Unit Tests]  │
└─────────────────────────────────────────────────────────────────┘

                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INTERACTION FLOW                             │
└─────────────────────────────────────────────────────────────────┘

    USER EXECUTES                 SCRIPT IMPORTS              TESTS VERIFY
         │                              │                           │
         ▼                              ▼                           ▼
┌──────────────────┐         ┌──────────────────┐       ┌──────────────────┐
│  make all        │────────▶│  scripts/01_*.py │◀──────│ tests/test_*.py  │
│  make test       │         │                  │       │                  │
└──────────────────┘         │  • Parse CLI args│       │  • Sample data   │
                             │  • Call src funcs│       │  • Mock inputs   │
                             │  • Print outputs │       │  • Assert outputs│
                             └────────┬─────────┘       └────────┬─────────┘
                                      │                          │
                                      │   import                 │   import
                                      ▼                          ▼
                             ┌─────────────────────────────────────┐
                             │         src/ modules                │
                             │                                     │
                             │  ✓ Pure functions (no side effects)│
                             │  ✓ Clear inputs/outputs            │
                             │  ✓ Testable in isolation           │
                             │  ✓ Reusable across scripts         │
                             └─────────────────────────────────────┘
```

---

## Scripts vs Tests Workflow

Understanding the difference between production scripts and test suite:

```
┌─────────────────────────────────────────────────────────────────┐
│                   TWO SEPARATE WORKFLOWS                         │
└─────────────────────────────────────────────────────────────────┘

    ANALYSIS PIPELINE                    TEST SUITE
    (Production)                         (Quality Assurance)
         │                                      │
         │                                      │
    ┌────▼─────┐                          ┌────▼─────┐
    │ make all │                          │make test │
    └────┬─────┘                          └────┬─────┘
         │                                      │
         ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐
│ Run Scripts     │                    │ Run Tests       │
│ 01-05           │                    │ test_*.py       │
└────┬────────────┘                    └────┬────────────┘
     │                                       │
     │ import                                │ import
     ▼                                       ▼
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│        src/ functions            │   │        src/ functions            │
│  ✓ Same code tested              │◀──│  ✓ Same code tested              │
└────┬─────────────────────────────┘   └────┬─────────────────────────────┘
     │                                       │
     │ use                                   │ use
     ▼                                       ▼
┌──────────────────┐                  ┌──────────────────┐
│ REAL DATA        │                  │ SAMPLE DATA      │
│ • UCI dataset    │                  │ • 4-100 rows     │
│ • 45k+ rows      │                  │ • Mock fixtures  │
│ • Saved outputs  │                  │ • Temp dirs      │
└────┬─────────────┘                  └────┬─────────────┘
     │                                      │
     ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐
│ RESULTS          │                  │ TEST RESULTS     │
│ • data/          │                  │ • Pass/Fail      │
│ • results/       │                  │ • Coverage: 78%  │
│ • reports/       │                  │ • No artifacts   │
└──────────────────┘                  └──────────────────┘

     KEEPS FILES                          CLEANS UP
     (for analysis)                       (after each test)

┌─────────────────────────────────────────────────────────────────┐
│                         KEY DIFFERENCES                          │
└─────────────────────────────────────────────────────────────────┘

  SCRIPTS (make all)              │  TESTS (make test)
  ───────────────────────────────────────────────────────────────
  ✓ Downloads 45k+ real records  │  ✓ Uses 4-100 sample rows
  ✓ Takes minutes to run         │  ✓ Completes in seconds
  ✓ Creates persistent files     │  ✓ Uses temp dirs (auto-deleted)
  ✓ Generates ML model & reports │  ✓ Verifies function logic
  ✓ Run when doing analysis      │  ✓ Run before commits
  ✓ Invoked: make all            │  ✓ Invoked: make test

┌─────────────────────────────────────────────────────────────────┐
│                    THEY NEVER RUN TOGETHER                       │
│                                                                   │
│  Running scripts does NOT trigger tests                          │
│  Running tests does NOT affect your data/results                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Developer Workflow

Docker image building and deployment workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Developer Makes Changes                       │
│                                                                   │
│  Edit any of:                                                    │
│  • environment.yml                                               │
│  • Dockerfile                                                    │
│  • conda-lock.yml                                                │
│  • .github/workflows/docker-publish.yml                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ git push
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              GitHub Actions: docker-publish.yml                  │
│                                                                   │
│  1. Update conda-lock.yml (if needed)                            │
│  2. Build multi-platform Docker image                            │
│     • linux/amd64                                                │
│     • linux/arm64                                                │
│  3. Push to Docker Hub with tags:                                │
│     • charlene1010/term-deposit-predictor:latest                 │
│     • charlene1010/term-deposit-predictor:<commit-sha>           │
│     • charlene1010/term-deposit-predictor:<branch-name>          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ on completion
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              GitHub Actions: run-analysis.yml                    │
│                                                                   │
│  1. Pull the newly built Docker image                            │
│  2. Run analysis scripts inside container                        │
│  3. Generate reports/outputs                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ analysis complete
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Developer Updates docker-compose.yml            │
│                                                                   │
│  1. Copy new image SHA from GitHub Actions logs                  │
│     Example: 4509ab91300d6d7725548bd407dca8958057c4a3           │
│                                                                   │
│  2. Update docker-compose.yml:                                   │
│     image: charlene1010/term-deposit-predictor:<new-sha>         │
│                                                                   │
│  3. Commit and push:                                             │
│     git add docker-compose.yml                                   │
│     git commit -m "Update Docker image to <new-sha>"             │
│     git push                                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ push to GitHub
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Team Members Pull Changes                     │
│                                                                   │
│  1. git pull                                                     │
│  2. docker compose up                                            │
│     • Automatically pulls new image with specific SHA            │
│     • Everyone uses exact same environment                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

This project follows best practices for:
- **Modularity**: Separation of concerns (src/ → scripts/ → tests/)
- **Testability**: Pure functions tested in isolation
- **Reproducibility**: Docker + conda-lock for consistent environments
- **CI/CD**: Automated builds and deployments via GitHub Actions
- **Quality Assurance**: Comprehensive test suite 
