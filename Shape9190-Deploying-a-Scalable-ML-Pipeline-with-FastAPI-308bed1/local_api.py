import json
import requests


url =  "http://127.0.0.1:8000"
response = requests.get(url)

# print the status code
print({"Status Code": response.status_code})
# print the welcome message
print(response.json())



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# send a POST using the data above
response = requests.post(url+"/data/", data=json.dumps(data))


# print the status code
print({"Status Code": response.status_code})
# print the result
print(response.json())
