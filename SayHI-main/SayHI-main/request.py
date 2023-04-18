import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'exp':1.8,})
print(r.json())


# Save the voice file as 0, 1, 2 etc. 
ans = {
    "0": "Male",
    "1": "Female",
    "2": "Female",
    "3": "Male"
}