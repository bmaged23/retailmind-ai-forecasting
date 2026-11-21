# RetailMind reports

These reports preserve the interactive analyses and project guides. GitHub displays HTML files as source; clone or download the repository and open the HTML file in a browser. The two data dashboards include their visualization libraries and work locally. The smaller guide pages describe the project rather than running model inference.

## Data and features

- [Data exploration dashboard](data-exploration.html): transaction history, sales patterns and data-quality findings. Supporting guide: [data acquisition](../data/README.md).
- [Feature analysis dashboard](feature-eda.html): historical product and transaction features. Supporting guide: [feature inventory](../docs/feature-inventory.md).
- [Product-code audit](code-family-audit.html): case variants and potential product families. Supporting guide: [feature inventory](../docs/feature-inventory.md).

## Forecast evaluation and decisions

- [Model comparison dashboard](comparison/validation-comparison.html): common-cohort model-selection results. [Evaluation README](../docs/evaluation/README.md) · [scorecard](comparison/scorecard.json).
- [Evaluation guide](evaluation.html): protocol, metrics and limitations. [Evaluation README](../docs/evaluation/README.md).
- [Uncertainty dashboard](milestone2-uncertainty.html): daily interval coverage and width. [Uncertainty methodology](../docs/milestone2-uncertainty.md).
- [Inventory simulation](milestone3-inventory.html): forecast-based replenishment under synthetic stock and supplier assumptions. [Inventory methodology](../docs/milestone3-inventory.md).
- [Scenario analysis](milestone4-scenarios.html): paired simulations of demand and delivery scenarios. [Scenario methodology](../docs/milestone4-scenarios.md).

## Service and operations

- [API guide](api.html) · [API README](../docs/api/README.md) · [endpoint reference](../docs/forecast-api.md).
- [Docker guide](docker.html) · [deployment README](../docs/docker/README.md) · [Docker Hub](https://hub.docker.com/r/bmaged23/retailmind-ai-forecasting).
- [CI/CD guide](cicd.html) · [CI/CD README](../docs/cicd/README.md) · [GitHub Actions workflow](../.github/workflows/docker-image-ci.yml).

Historical accuracy and simulation results are not evidence of current production performance or realized savings. Read the linked methodologies before drawing conclusions.
