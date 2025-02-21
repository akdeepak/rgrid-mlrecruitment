import json
import requests


BASE_URL = "http://127.0.0.1:5000/"

# You can use this function to test your api
# Make sure the server is running locally on BASE_URL`
# or change the hardcoded localhost above
def test_predict():
    """
    Test the predict route with test data
    """
    long_text = "This is a long document that exceeds the token limit of 512. " * 50

    # test_description = {"description": "this is a test description about Dementia"}

    # test_description = {"description": "This is a long document that exceeds the token limit of 512. " * 50}
    print("Calling API with test description:")
    test_description = {"description": "With the development of supportive measures, the natural history of ALS has changed. Researchers compared the natural history of ALS patients from 1999-2004 and 1984-1998 and found that the median survival time was significantly longer in the former than in the latter (4.32 years vs. 3.22 years) and that the disease progression was slower in the former, even after adjusting for other confounding factors. Although previous studies have provided reference for the diagnosis and treatment of ALS, the etiology of ALS is still unknown, and the relevant clinical features and natural history of ALS still lack the verification of large samples. Therefore, the research on the natural history of ALS is of great significance to further increase the understanding of ALS and provide new evidence for the diagnosis and treatment of ALS."}
    print("Calling API with test description:")
    # response = requests.post(f"{BASE_URL}/predict", data=json.dumps(test_description))
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
    test_predict()
    # test_generate_eda()
    #test_train()
