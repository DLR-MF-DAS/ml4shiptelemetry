# ML4ShipTelemetry

## Framework

This is a collaboration between the Department of Data Science ([MF-DAS](https://www.dlr.de/de/eoc/ueber-uns/institut-fuer-methodik-der-fernerkundung/eo-data-science)) at the Earth Observation Center (EOC) of German Aerospace Center (DLR) and [Geomar](https://www.geomar.de/), within the [Helmholtz AI consulting framework](https://www.helmholtz.ai/you-helmholtz-ai/ai-consulting/).

## Goal

The goal of this project is to build a Machine Learning (ML) model capable of inferring data typically calculated through mathematical methods from thermosalinograph data collected during ship expeditions. Specifically, we aim to determine whether an ML model can accurately reconstruct temperature and salinity trends using multivariate time series data from ship sensors, as well as predict data quality-related flags.

## Methodology

We approached this task as a supervised learning problem, with thermosalinograph measurements as input data and temperature and salinity as output targets. Using temperature and salinity ground truth data provided by GEOMAR, we trained a Random Forest (RF) regressor. We performed 5-fold cross-validation to evaluate the model, measuring its accuracy with the R² score, which indicates the percentage of variance in the dependent variable explained by the model. Our RF regressor achieved an R² score of 0.89 for temperature and 0.96 on salinity, and visual analysis of the predictions confirms that the model's output closely aligns with the reference data. The test set included geographical regions not covered by the training data and therefore indicate a good ability in the model to generalize to previously unvisited regions.

| Temp Reference         | Temp Prediction         | Sal Reference         | Sal Prediction         |
|------------------------|-------------------------|-----------------------|------------------------|
| ![Temp Reference](imgs/temp_ref.jpg) | ![Temp Prediction](imgs/temp_pred.jpg) | ![Sal Reference](imgs/sal_ref.jpg) | ![Sal Prediction](imgs/sal_pred.jpg) |

In the classification problem of predicting data quality flags, the model achieves a balanced accuracy of 81% on the temperature flags and 93% on the salinity flags. Balanced accuracy is used due to the high degree of imbalance between the good and bad quality flags, roughly 90% and 10% of the samples, respectively. A feature importance study highlights the importance of keeping a steady flow through the measurement containers, as either too high or to low flow is related to higher propensity of bad data quality.

## Run Our Work

Our project is packaged as a Python module, which can be easily installed with:

```bash
pip install ml4shiptelemetry
```

Once installed, you can train and evaluate the models on the training set with:

```bash
python -m ml4shiptelemetry --data-dir your/path/to/data
```

To evaluate on a test set, set the program to use a given number of data files (3 in this example) as test:
```bash
python -m ml4shiptelemetry --data-dir your/path/to/data --n-test-files 3
```

Once installed, you can begin cross-validating the RF regressor with:

```bash
python -m ml4shiptelemetry --data-dir your/path/to/data --cv
```

Replace `your/path/to/data` with the directory containing your thermosalinograph and ground truth data.

For datasets with time series nature, i.e. the samples are ordered and subsequent samples are correlated, a time series cross validation can be used instead. Activate the time series cross validation by using the `--ts_cv` flag:

```bash
python -m ml4shiptelemetry --data-dir your/path/to/data --cv --ts_cv
```

The following flags can be used in the command line call:

| Command | Type | Description |
| ------- | ---- | ----------- |
| `--data-dir` | string | Path to raw data. Required. |
| `--cv ` || Crossvalidate model performance on training data. Leave out to not crossvalidate. Applies K-fold without shuffling (unless overridden by ts-cv). |
| `--ts-cv` || Use time series cross validation instead of regular cross validation. If included, overrides --cv. |
| `--cv-params` | string | Path to json file containing cross-validation hyperparameters to exhaustively evaluate. See example file cv_params.json for how to structure this file. |
| `--n-test-files` | integer | Number of data files to use as test set. Test files are picked from the back of the list of files. |
| `--n-neighbours` | integer | Number of neighbouring rows to add to each row in the training set. For example, a value 1 means adding the row before and after. |
| `--preprocessed-dir` | string | Directory to store processed data for faster reprocessing next time. Defaults to a folder data/ directly under the main folder of the repository. |
| `--model-output-dir` | string | Directory to store models as pickle files. Defaults to a folder output/ directly under the main folder of the repository. |
| `--log-dir` | string | Directory to save log file. Defaults to a folder logs/ directly under the main folder of the repository. Both training and evaluation output are logged. |
| `--verbose` | integer | Verbosity of scikit-learn GridSearchCV. |


## Explainable AI analysis

An analysis of the features contributing to the prediction outcomes, utilizing SHAP, is included under `analysis/xai_analysis.ipynb`

## Future work

We want to extend what we did until now by:
- Using statistical procedures to add confidence intervals to the prediction
- Exploring the use of deep learning methods
