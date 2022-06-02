# Price Suggestion

This is the code for a machine learning model that suggests prices from listing information.

## Requirements

Python 3.7 or higher  
Docker

## Getting Started

Install the dependencies:
```bash
pip install -r requirements.txt
```
## Usage

Activate the virtual environment and start the Flask application:
```bash
. venv/bin/activate
python api/app.py
```

Run the unit tests:
```bash
python api/test_app.py
```

Send a request:
```bash
python request.py --row 0
```

To build the Docker image, create and start a container:
```bash
docker build --tag predict-price .
docker run predict-price
```

I tested the project on macOS 12.3.1.

## Goal

On online thrift stores, users can choose any price they want when listing an item which can cause the items to have higher than market prices and unsold items. On the other hand, if the listing price is lower than the market price, the Mercari customers lose money. As a solution, users can search the Mercari database for an item they plan to list. But not only is this a lot of effort but also this is tricky for new Mercari users. Therefore, listing becomes easy if we automatically display a suitable price for users when they list an item.

## Data Analysis

First, I did an exploratory data analysis (see `exploratory_analysis.ipynb`). I checked the missing values and unique values in each column. I visualized the most frequent values for names, brand names, category names, and item descriptions. Then I checked the distribution of item condition, shipment, and price variables. The item condition id has a negative relationship with the price. The item description of the train and test datasets are similar in terms of length. The item description length is not strongly correlated with the mean or median price but has a positive trend.

## Training

The training process is in the `train_validate_predict.ipynb`. For preprocessing, I converted the text columns to token counts. I processed the categorical columns by binarizing them. For training, I first used gradient boosting with XGBoost. Then I tried a fully connected neural network with TensorFlow. I got better results with XGBoost, so I used the model for prediction.

## Results

|   Model    | Training RMSLE | Validation RMSLE | Training MAE | Validation MAE |
|------------|:--------------:|:----------------:|:------------:|:--------------:|
|  XGBoost   |     0.4187     |      0.4995      |    7.4214    |     9.4769     |
| TensorFlow |     0.4534     |      0.5144      |    8.4607    |     9.9160     |
