---
title: 'zTSkut: A web application for predicting thermoelectric performance in skutterudites with deep learning'
tags:
  - Python
  - thermoelectrics
  - skutterudites
  - machine learning
  - deep learning
  - FastAPI
authors:
  - name: Victor Posligua
    orcid: 0000-0003-3375-3706
    affiliation: 1
    corresponding: true
  - name: Karina Landivar
    orcid: 0009-0007-0160-1448
    affiliation: 1
  - name: Elena R. Remesal
    orcid: 0000-0001-6984-0647
    affiliation: 1
  - name: Gerda Rogl
    orcid: 0000-0002-8056-5006
    affiliation: 2
  - name: Peter F. Rogl
    orcid: 0000-0002-7733-1612
    affiliation: 2
  - name: Javier Fdez Sanz
    orcid: 0000-0003-2064-7007
    affiliation: 1
  - name: Jesus Prado-Gonjal
    orcid: 0000-0003-4880-8503
    affiliation: 3
  - name: Antonio M. Márquez
    orcid: 0000-0001-6699-064X
    affiliation: 1
  - name: Jose J. Plata
    orcid: 0000-0002-0859-0450
    affiliation: 1
    corresponding: true
affiliations:
  - name: Departamento de Química Física, Facultad de Química, Universidad de Sevilla, Seville, Spain
    index: 1
  - name: Institute of Materials Chemistry, University of Vienna, Waehringerstraße 42, Wien, Austria
    index: 2
  - name: Departamento de Química Inorgánica, Universidad Complutense de Madrid, Madrid, Spain
    index: 3
date: 22 May 2026
bibliography: paper.bib
---

# Summary

zTSkut is a Python-based web application for predicting the thermoelectric figure of merit, zT, of skutterudite-based compositions. The software provides a browser interface for single-composition prediction and a CSV-upload workflow for batch screening. Users specify the anion, cation and filler species, their corresponding fractions, carrier concentration at 300 K, temperature and carrier type. zTSkut then generates the same compositional descriptors used by the trained neural-network model, applies the saved feature scaling from the original training workflow and returns predicted zT values.

The software is built around a FastAPI backend and a lightweight HTML/JavaScript frontend. The backend loads a trained Keras model, a saved `StandardScaler`, and the descriptor order expected by the model. This design makes the prediction workflow reproducible: features are transformed using the original training scaler in transform-only mode and no fitting or re-normalisation is performed during user prediction. zTSkut also supports input validation to catch common formatting errors, such as placing element symbols in numerical fraction fields.

# Statement of need

Skutterudites are an important family of thermoelectric materials because their structure allows chemical tuning through filler, cation and anion substitution. This large compositional flexibility makes them suitable for computational screening, but it also creates a practical barrier for experimental researchers: evaluating many candidate compositions using first-principles transport calculations or synthesis is expensive and slow. Machine learning (ML) models can help prioritise promising candidates, but they are only useful to the broader community if they can be accessed without requiring users to reproduce the full training and descriptor-generation workflow.

zTSkut addresses this need by exposing a trained skutterudite zT model through a simple web interface. The underlying neural-network model and its scientific validation are described in Posligua et al. [@posligua2026skutterudites]. The purpose of zTSkut is not to replace that model-development study, but to make the trained predictor usable for screening new skutterudite candidates. The web app allows users to test individual compositions interactively or upload large candidate lists for batch prediction. This is particularly useful for researchers designing filled or multi-filled skutterudites and looking for a rapid first-pass estimate of thermoelectric performance before more expensive experimental or computational validation.

# State of the field

Several ML models have been developed for thermoelectric materials, including models trained on broad thermoelectric datasets or on specific material families. However, these models are often published primarily as methodological studies, with limited user-facing software for applying the trained model to new candidate compositions. In practice, reusing such models can require reproducing the original descriptor generation, feature ordering, scaling and loading workflow, which creates a barrier for experimental or computational users who simply want to screen new compositions.

General purpose ML platforms and Python libraries such as TensorFlow/Keras [@tensorflow; @keras] and scikit-learn [@scikit-learn] provide the underlying infrastructure for training and deploying models, but they do not provide a domain-specific interface for skutterudite zT prediction. Similarly, web frameworks such as FastAPI [@fastapi] make deployment possible, but they do not encode the thermoelectric descriptor logic, saved model artefacts or input conventions required for this specific skutterudite predictor.

zTSkut therefore fills a focused gap: it packages the trained skutterudite model from Posligua et al. [@posligua2026skutterudites] into a reproducible and user-facing prediction tool. Rather than contributing to a general platform, zTSkut was built as a lightweight dedicated application because the model requires a specific descriptor representation, a fixed feature order and the saved training scaler to ensure that predictions are consistent with the peer-reviewed model development workflow. This narrow scope is intentional: it keeps the tool easy to use, easy to deploy and directly aligned with the skutterudite screening problem.

# Software design

The main design goal of zTSkut is to make the trained skutterudite zT model accessible while preserving the exact preprocessing workflow used during model development. For this reason, the software separates the user interface from the prediction pipeline but keeps the deployment structure simple enough to run locally or on a lightweight web service.

The backend is implemented with FastAPI. The `/predict` endpoint receives either a single-system form converted internally into a CSV file or an uploaded batch CSV file. The prediction script then reads the input, validates numerical fields, replaces empty optional fields with zero, generates compositional descriptors, orders them according to `feature_columns_ZT.txt`, applies the saved `scaler_ZT.joblib` and finally evaluates the trained `model_keras_skutt.h5` model. This design avoids fitting any scaler or model component during prediction, reducing the risk of data leakage or inconsistent preprocessing.

A deliberate trade-off was made to keep the software lightweight and transparent rather than building a more complex package architecture. The current implementation uses a small number of files so that it can be deployed easily on Render and run locally with a single Uvicorn command. This is important for the intended audience: researchers who want to use the model for candidate screening should not need to install a large framework or understand the full training workflow. At the same time, the repository includes tests for basic web app functionality, prediction through the endpoint and input validation, so that reviewers and users can verify that the core workflow is functioning.

The frontend is implemented as a lightweight HTML/CSS/JavaScript interface without additional frontend framework dependencies. The interface supports both single-composition prediction and batch prediction, as these represent two common use cases: quickly checking one proposed composition and screening many candidates generated by enumeration or chemical intuition.

# Software functionality

The current implementation provides the following functionality:

- Single-system prediction through a browser form
- Batch prediction from CSV files
- Automatic treatment of empty optional composition and fraction fields as zero
- Generation of the compositional descriptor set expected by the trained model
- Application of the saved training scaler and fixed feature order
- Input validation for numerical fields such as fractions, temperature, carrier concentration and carrier type
- Downloadable citation files for users of the model
- Local execution through FastAPI as well as deployment as a web app

The main user-facing workflow is intentionally simple. For single predictions, users fill in the composition fields and submit the form. For batch predictions, users download the CSV template, add one candidate composition per row and upload the completed file. The app returns one predicted zT value per row.

# Research impact statement

The scientific model served by zTSkut has already been peer reviewed and published in Journal of Materials Chemistry A [@posligua2026skutterudites]. In that study, the model was trained on a curated skutterudite dataset and evaluated using internal validation, independent external testing, comparison with experimental trends and physical interpretation through SHAP analysis and first-principles calculations. zTSkut converts that published model into a practical software tool that can be used directly by other researchers.

The near-term impact of zTSkut is to make skutterudite screening accessible without requiring users to reproduce the original data-processing and loading workflow. This is especially relevant for experimental groups considering new filled or multi-filled skutterudite compositions, because the tool can provide a rapid first-pass estimate before synthesis or expensive transport calculations. The app also supports batch prediction, allowing users to screen large candidate lists rather than testing compositions one by one.

The software has been prepared for community use in several practical ways: it is available as a live web app, includes a documented CSV input format, provides citation files, supports local execution, and includes tests covering the web endpoint and input validation. These features are intended to make the model reusable beyond the original manuscript and to support future integration into broader thermoelectric screening workflows.

# Availability and use

The web application is available at <https://ztskut.onrender.com/>. The repository includes a CSV template for batch prediction, citation files, and basic tests for the web endpoint and input validation. The software can also be run locally with Uvicorn using the command:

```bash
uvicorn main:app --reload
```

zTSkut is intended as a screening and prioritisation tool. Predictions should be interpreted within the chemical and thermoelectric domain represented by the model-development dataset. In particular, predictions for compositions far outside the training chemistry, unusual carrier concentrations or temperatures, systems affected by secondary phases, or very high-zT regimes should be treated with caution and validated experimentally or with higher-fidelity calculations.

# AI usage disclosure

No AI tools were used to assist with the code and web app. All scientific content, software behaviour, tests and final text were reviewed, edited, and validated by the authors. The trained model, dataset, scientific interpretation, and research conclusions are based on the peer-reviewed work by Posligua et al. [@posligua2026skutterudites].

# Acknowledgements

The authors acknowledge the collaborative work underlying the original skutterudite dataset, model development and validation reported in [@posligua2026skutterudites]. Do we need to add the same acknowledgements like in the main paper?
