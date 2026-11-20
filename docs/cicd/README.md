# CI/CD pipeline

GitHub Actions builds the image, runs real inference and decision API smoke tests, then publishes the tested image.

## Triggers

Pull requests, pushes to main, and manual dispatch run the build and smoke test.

## Quality gate

The workflow starts the container and runs scripts/ops/smoke_test_container.py. Container logs are shown on failure.

## Publish

After success on main, Docker Hub login uses DOCKERHUB_USERNAME and DOCKERHUB_TOKEN repository secrets; the tested image is tagged latest and with the Git commit SHA.

## Current status

The workflow is prepared in the repository. A GitHub Actions run and Docker Hub publication must be confirmed after secrets are configured; local smoke testing does not prove the hosted pipeline passed.

[Detailed reference](../../.github/workflows/docker-image-ci.yml) · [Visual guide](../../reports/cicd.html) · [Project README](../../README.md)
