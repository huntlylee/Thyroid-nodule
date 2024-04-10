
# Thyroid Nodule Risk Prediction

![GitHub License](https://img.shields.io/github/license/huntlylee/Thyroid-nodule)

This README provides a tutorial for using a Jupyter Notebook to predict the risk of developing thyroid nodules based on user input. The notebook utilizes pre-trained machine learning models to assess the 3-year risk of nodule onset.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or above
- Jupyter Notebook
- Required Python libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `xgboost`, `lightgbm`, `joblib`

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib
pip install xgboost
pip install lightgbm
pip install joblib
```

## Getting Started

1. Clone the repository or download the Jupyter Notebook '''Tutorial.ipynb''' and the pre-trained models.
2. Launch Jupyter Notebook by running the following command in your terminal:

```bash
jupyter notebook
```

3. Navigate to the directory containing the downloaded notebook and open it.

## Notebook Structure

The notebook is structured as follows:

1. **Import**: Import all required packages.

2. **Data Input**: A section for inputting user data. Please enter all required clinical features, such as age, HDL-C levels, FBG levels, and creatinine levels, into the proper fields. You may also select from different pre-trained machine learning models.

3. **Data Preprocessing**: To automatically preprocess the above user inputs to the standard data format recognized by the model

4. **Model Loading**: Code to load the pre-trained machine learning model.

5. **Risk Prediction**: The notebook will use the input data to predict the risk of developing thyroid nodules and display the result.

## Using the Notebook

Follow the instructions within the notebook to input the required data. Each cell can be executed by selecting it and pressing `Shift + Enter`. Ensure that you run the cells in the order they appear.

## Alternative Tools

* A calibrated [nomogram](Nomogram.png) based on the regression model is provided to simplify the risk assessment in clinical settings.

![Alternative Tools](Nomogram.png)

* ${\textsf{\color{green}An Excel spreadsheet}}$ ```Thyroid nodule risk prediction.xlsx``` is also available to facilitate the use of this approach on electronic devices. Users just need to enter values for required clinical indicators, and the risk can be automatically calculated.

![Alternative Tools](Excel%20tool.png)

## Support

For any issues or questions regarding the notebook, please contact the repository maintainer.

## Data Privacy

Please note that any input data you provide should be handled in accordance with relevant data privacy regulations. Do not share personal health information in public repositories or forums.

## License

This tutorial and the accompanying Jupyter Notebook are provided under the [Apache 2.0 license](LICENSE).

## Acknowledgments

We would like to thank the contributors and maintainers of the Python libraries used in this project.

