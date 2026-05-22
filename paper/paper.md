---
title: 'zTSkut: A web application for machine-learning prediction of thermoelectric performance in skutterudites'
tags:
  - Python
  - thermoelectrics
  - skutterudites
  - machine learning
  - materials informatics
  - FastAPI
authors:
  - name: Victor Posligua
    affiliation: 1
    corresponding: true
affiliations:
  - name: Institute for Functional Intelligent Materials, National University of Singapore, Singapore
    index: 1
date: 22 May 2026
bibliography: paper.bib
---

# Summary

zTSkut is a Python-based web application for predicting the thermoelectric figure of merit, zT, of skutterudite-based compositions. The software provides a browser interface for single-composition prediction and a CSV-upload workflow for batch screening. Users specify the anion, cation and filler species, their corresponding fractions, carrier concentration at 300 K, temperature, and carrier type. zTSkut then generates the same compositional descriptors used by the trained neural-network model, applies the saved feature scaling from the original training workflow, and returns predicted zT values.

The software is built around a FastAPI backend and a lightweight HTML/JavaScript frontend. The backend loads a trained Keras model, a saved `StandardScaler`, and the descriptor order expected by the model. This design makes the prediction workflow reproducible: features are transformed using the original training scaler in transform-only mode, and no fitting or re-normalisation is performed during user prediction. zTSkut also supports input validation to catch common formatting errors, such as placing element symbols in numerical fraction fields.

# Statement of need

Skutterudites are an important family of thermoelectric materials because their structure allows chemical tuning through filler, cation and anion substitution. This large compositional flexibility makes them suitable for computational screening, but it also creates a practical barrier for experimental researchers: evaluating many candidate compositions using first-principles transport calculations or synthesis is expensive and slow. Machine-learning models can help prioritise promising candidates, but they are only useful to the broader community if they can be accessed without requiring users to reproduce the full training and descriptor-generation workflow.

zTSkut addresses this need by exposing a trained skutterudite zT model through a simple web interface. The underlying neural-network model and its scientific validation are described in Posligua et al. [@posligua2026skutterudites]. The purpose of zTSkut is not to replace that model-development study, but to make the trained predictor usable for screening new skutterudite candidates. The web app allows users to test individual compositions interactively or upload large candidate lists for batch prediction. This is particularly useful for researchers designing filled or multi-filled skutterudites and looking for a rapid first-pass estimate of thermoelectric performance before more expensive experimental or computational validation.

# Software functionality

The current implementation provides the following functionality:

- single-system prediction through a browser form;
- batch prediction from CSV files;
- automatic treatment of empty optional composition and fraction fields as zero;
- generation of the compositional descriptor set expected by the trained model;
- application of the saved training scaler and fixed feature order;
- input validation for numerical fields such as fractions, temperature, carrier concentration and carrier type;
- downloadable citation files for users of the model;
- local execution through FastAPI as well as deployment as a web app.

The main user-facing workflow is intentionally simple. For single predictions, users fill in the composition fields and submit the form. For batch predictions, users download the CSV template, add one candidate composition per row, and upload the completed file. The app returns one predicted zT value per row.

# Implementation

zTSkut is implemented in Python using FastAPI [@fastapi] for the backend web service. The prediction workflow uses TensorFlow/Keras [@tensorflow; @keras] to load the trained neural-network model and scikit-learn [@scikit-learn] to apply the saved feature scaler. Data handling is performed using pandas [@pandas] and NumPy [@numpy]. The frontend is written in plain HTML, CSS and JavaScript to keep the deployment lightweight and easy to maintain.

The deployed version uses a compact repository structure in which `main.py` defines the FastAPI routes and `load_model_predictions_server.py` performs input parsing, descriptor generation, feature ordering, scaling and prediction. The files `model_keras_skutt.h5`, `scaler_ZT.joblib` and `feature_columns_ZT.txt` define the trained model, saved scaler and required descriptor order, respectively.

# Availability and use

The web application is available at <https://ztskut.onrender.com/>. The repository includes a CSV template for batch prediction, citation files, and basic tests for the web endpoint and input validation. The software can also be run locally with Uvicorn using the command:

```bash
uvicorn main:app --reload
```

zTSkut is intended as a screening and prioritisation tool. Predictions should be interpreted within the chemical and thermoelectric domain represented by the model-development dataset. In particular, predictions for compositions far outside the training chemistry, unusual carrier concentrations or temperatures, systems affected by secondary phases, or very high-zT regimes should be treated with caution and validated experimentally or with higher-fidelity calculations.

# Generative AI disclosure

Generative AI tools were used to assist with code refactoring, documentation drafting, and manuscript editing. All software design decisions, scientific content, code validation, and final text were reviewed and approved by the author.

# Acknowledgements

The trained model used by zTSkut was developed as part of the skutterudite thermoelectrics study reported by Posligua et al. [@posligua2026skutterudites]. The author thanks the coauthors of that work for the scientific collaboration underlying the model and dataset.
