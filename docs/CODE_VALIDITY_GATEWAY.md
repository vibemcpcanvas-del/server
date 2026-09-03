# Code Validity Gateway

## Purpose

This repository treats GitHub Actions as the code validity gateway for changes that may be provided to users or merged into `main`.

A change is not considered ready for use merely because it was written or reviewed. It must pass the automated validation gates defined in the repository before it can be recommended for local, Colab, training, evaluation, or production use.

## Required flow

```text
Change proposed
    -> isolated branch
    -> pull request
    -> GitHub Actions validation gateway
    -> passing checks
    -> review and merge to main
    -> downstream training, evaluation, or deployment
```

Do not instruct users to run unvalidated code from a branch, chat message, notebook, or local workaround.

## Required validation gates

Every pull request targeting `main` must run and pass the following checks:

1. Python compilation for project source, scripts, and tests.
2. Core module import checks, including the Jin Hilla scenario environment.
3. Environment smoke tests that construct the relevant environment and call its basic initialization path.
4. Unit and regression tests.
5. Credential scanning to prevent secrets from being committed.

A failed gate blocks the change from being treated as valid. The failure must be fixed in code and validated again; bypassing a failing check with a local manual edit is not an acceptable release path.

## M8 policy

M8 training and evaluation must use only a commit that has passed the validation gateway. Before issuing a Colab command for M8:

- Confirm the relevant pull request checks are green.
- Use the validated `main` commit after merge, or explicitly identify a validated branch commit for controlled testing.
- Do not use an earlier checkpoint as evidence that newly changed code is valid.

## Regression prevention

When an incident reveals a missing import, incompatible interface, runtime failure, or incorrect output contract:

1. Add or strengthen an automated test that reproduces the incident.
2. Fix the underlying code rather than relying on an execution-time workaround.
3. Require the regression test to pass in the gateway for all future pull requests.

For example, an import failure involving `JinHillaEnvironmentCore` must be protected by a test that imports the exact public symbols used by the scenario environment and its tests.

## Branch protection

Repository administrators should configure `main` branch protection to require the validation workflow(s) to pass before merge and to prevent direct pushes that bypass pull-request checks.

## Operating rule

Green GitHub Actions checks are the minimum acceptance signal. A green result confirms the checks ran successfully; it does not replace domain review, training-quality evaluation, or release approval where those are required.
