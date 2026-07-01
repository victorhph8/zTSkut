# Contributing to zTSkut

Thank you for your interest in zTSkut.

zTSkut is a focused research web application for predicting the thermoelectric figure of merit, zT, of skutterudite-based compositions using the trained model reported by Posligua et al. The scope of this repository is to make the published model accessible through a reproducible web interface and local prediction workflow.

## Reporting issues

If you find a problem, please open a GitHub issue and include:

* A short description of the problem
* The input file or input values that produced the issue, if possible
* The expected behaviour
* The observed behaviour
* Any relevant error message

Please avoid sharing confidential or unpublished experimental data in public issues.

## Suggesting improvements

Suggestions are welcome if they are aligned with the scope of the project. Useful suggestions may include:

* Improving input validation or error messages
* Clarifying documentation
* Adding examples or tests
* Improving the web interface
* Reporting cases where the software behaves unexpectedly

Large changes to the model architecture, training dataset or scientific methodology are outside the current scope of this repository unless discussed first.

## Code contributions

Code contributions can be proposed through pull requests. Before making a large change, please open an issue to discuss the motivation and intended behaviour.

Contributions should preserve the existing prediction workflow:

1. Read the user input
2. Validate numerical fields
3. Generate descriptors
4. Align features with `feature_columns_ZT.txt`
5. Apply `scaler_ZT.joblib`
6. Evaluate `model_keras_skutt.h5`
7. Return predicted zT values

Please also update or add tests when changing behaviour.

## Support expectations

This is academic research software maintained on a best-effort basis. Issues and contributions will be reviewed when time allows.
