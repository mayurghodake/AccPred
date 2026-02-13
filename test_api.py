import requests

# Test the status endpoint
try:
    response = requests.get('http://localhost:5000/api/status')
    print(f"Status endpoint response: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error testing status endpoint: {e}")

# Test the analyze endpoint with a dummy file
try:
    with open('sample_traffic.mp4', 'rb') as f:
        files = {'video': f}
        response = requests.post('http://localhost:5000/api/analyze', files=files)
        print(f"Analyze endpoint response: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error testing analyze endpoint: {e}")
