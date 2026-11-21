# CI/CD pipeline

GitHub Actions builds the image, runs real inference and decision API smoke tests, then publishes the tested image.

## Triggers

Pull requests, pushes to main, and manual dispatch run the build and smoke test.

## Quality gate

The workflow starts the container and runs scripts/ops/smoke_test_container.py. Container logs are shown on failure.

## Publish

After success on main, Docker Hub login uses DOCKERHUB_USERNAME and DOCKERHUB_TOKEN repository secrets; the tested image is tagged latest and with the Git commit SHA.

## Current status

GitHub Actions has completed a hosted build and real-model smoke test successfully. The initial verified image was pushed to Docker Hub separately. Automated publication is skipped with a warning until both Docker Hub repository secrets are configured; a successful CI run alone does not confirm a new image was published.

[Detailed reference](../../.github/workflows/docker-image-ci.yml) · [Visual guide](../../reports/cicd.html) · [Project README](../../README.md)

## Visual guides

[CI/CD HTML guide](../../reports/cicd.html) · [Docker HTML guide](../../reports/docker.html) · [API HTML guide](../../reports/api.html) · [Report index](../../reports/README.md). Open the HTML files in a browser after downloading the repository.
