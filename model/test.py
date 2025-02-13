import requests

url = "http://127.0.0.1:5000/predict"
files = {"file": open("test_image.jpg", "rb")}  # Replace with your test image
response = requests.post(url, files=files)
print(response.json())  # Should return a prediction
