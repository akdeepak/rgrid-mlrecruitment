import json
import requests
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import nltk
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_URL = "http://127.0.0.1:5000/"

# You can use this function to test your api
# Make sure the server is running locally on BASE_URL`
# or change the hardcoded localhost above
def test_predict():
    """
    Test the predict route with test data
    """
    test_description = {"description": "this is a test description about Dementia"}
    print("Calling API with test description:")
    response = requests.post(f"{BASE_URL}/predict", data=json.dumps(test_description))
    print("Response: ")
    print(response.status_code)
    print(response.json())
    assert response.status_code == 200


def test_generate_eda():
    """
    Test the /generate_eda route with a sample CSV file.
    """
    test_data = {"file_path": "data/trials.csv"}
    
    print("Calling API with test CSV file...")
    response = requests.post(f"{BASE_URL}/generate_eda", data=json.dumps(test_data))
    
    print("Response:")
    print(response.status_code)
    print(response.json())

    assert response.status_code == 200
    assert "message" in response.json()
    assert "report_path" in response.json()


def test_train():
    """
    Test the /train route with a sample CSV file.
    """
    test_data = {"file_path": "data/trials.csv"}
    
    print("Calling API to train model...")
    response = requests.post(f"{BASE_URL}/train", json=test_data)

    print("Response:")
    print(response.status_code)
    print(response.json())

    assert response.status_code == 200
    assert "message" in response.json()
    assert "model_path" in response.json()

    
if __name__ == "__main__":
    # test_predict()
    #test_generate_eda()
    test_train()