# Packaged Timestral 209M LoRA adapter

This is the adapter available from the project's earlier July validation run.
It was trained through 2011-04-30 and evaluated on 2011-07-01 through
2011-07-28. The October test scorecard selected the same model family and
fine-tuning approach as its winner, but its adapter checkpoint was not retained.
Therefore the October test metrics must not be attributed to these exact weights.

The adapter requires the `mistralai/Timestral-209M` base model. The Docker build
downloads and packages that base model alongside this adapter. The API serves
only this model and returns daily P10/P50/P90 estimates plus horizon-wise
calibrated bounds. Inventory proposals and scenario simulations remain
available as decision-support operations.

The adapter, base model, and their derivatives are subject to the included
Mistral Modified MIT license, including its monthly revenue condition. See
`training-metadata.json` for the adapter's validation-run parameters and
`calibration.json` for the saved interval adjustments.
