from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
from sklearn.neighbors import KDTree
import requests
from fuzzywuzzy import process
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the data and prepare the model
print("Loading and preparing data...")
data = pd.read_csv('./data/pincode_offices.csv', sep=';', low_memory=False)
data.columns = data.columns.str.strip()
data['OfficeName'] = data['OfficeName'].astype(str).str.lower()
data['Pincode'] = data['Pincode'].astype(str).str.lower()
data['Pincode'] = pd.to_numeric(data['Pincode'], errors='coerce')
data.dropna(subset=['Pincode'], inplace=True)
data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
data.dropna(subset=['Latitude', 'Longitude'], inplace=True)

features = data[['Latitude', 'Longitude']].values
labels = data['OfficeName'].values
pincodes = data['Pincode'].values.astype(int)

# Initialize KDTree
tree = KDTree(features)
print("Data loaded and model prepared.")

def get_lat_long_from_osm(address):
    print(f"Fetching lat/long for address: {address}")
    url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json"
    response = requests.get(url)
    data = response.json()
    
    if data:
        lat = data[0]['lat']
        lon = data[0]['lon']
        print(f"Latitude and Longitude retrieved: {lat}, {lon}")
        return float(lat), float(lon)
    else:
        print("No data found for the given address.")
        return None, None

def suggest_similar_offices(input_office_name, input_pincode):
    print(f"Suggesting similar offices for: {input_office_name} with Pincode: {input_pincode}")
    
    similar_offices = data[data['Pincode'] == input_pincode][['OfficeName', 'Pincode']]
    
    if not similar_offices.empty:
        result = [(row['OfficeName'], row['Pincode']) for index, row in similar_offices.iterrows()]
        print(f"Similar offices found: {result}")
        return result
    
    all_office_names = data['OfficeName'].values
    suggestions = process.extract(input_office_name, all_office_names, limit=3)
    
    print(f"Fuzzy matching results: {suggestions}")
    return [(suggestion[0], data[data['OfficeName'] == suggestion[0]]['Pincode'].values[0]) for suggestion in suggestions]

def clean_office_name(office_name):
    # Strip known suffixes (e.g., "b.o", "s.o") and normalize the name
    suffixes = [" b.o", " s.o"]
    for suffix in suffixes:
        if office_name.endswith(suffix):
            return office_name[:-len(suffix)].strip()
    return office_name.strip()

def search_office(full_address):
    start_time = time.time()
    print(f"Searching for address '{full_address}'")
    
    *address_parts, input_pincode = full_address.rsplit(',', 1)
    input_pincode = input_pincode.strip()
    
    if not input_pincode.isdigit():
        print("Invalid Pincode found in the address. Please check the address format.")
        return "Invalid Pincode. Please check the address format."
    
    # Clean the input office name
    input_office_name = clean_office_name(' '.join(address_parts).lower().strip())
    print(f"Input Office Name: '{input_office_name}'")
    
    # Clean office names in the DataFrame for comparison
    data['CleanedOfficeName'] = data['OfficeName'].str.lower().apply(clean_office_name)
    
    # Check for an exact match based on cleaned values
    pincode_data = data[(data['CleanedOfficeName'] == input_office_name) & 
                        (data['Pincode'] == int(input_pincode))]
    
    if not pincode_data.empty:
        print("Exact match found in the dataset.")
        elapsed_time = time.time() - start_time
        return f"Exact match found: {input_office_name} in Pincode {input_pincode} in {elapsed_time:.2f} seconds"
    
    # If no exact match, suggest similar offices
    similar_offices = suggest_similar_offices(input_office_name, int(input_pincode))
    
    if similar_offices:
        similar_office_list = ', '.join([f"{office[0]} (Pincode: {int(office[1])})" for office in similar_offices])
        print("No exact match found. Providing suggestions.")
        elapsed_time = time.time() - start_time
        return f"No exact match found. Did you mean one of these offices?: {similar_office_list} in {elapsed_time:.2f} seconds"
    
    # Handle case where no exact match and no similar offices were found
    pincode_data = data[data['Pincode'] == int(input_pincode)]
    
    if not pincode_data.empty:
        latitude = pincode_data['Latitude'].values[0]
        longitude = pincode_data['Longitude'].values[0]
        print(f"Using latitude and longitude from dataset: {latitude}, {longitude}")
    else:
        latitude, longitude = get_lat_long_from_osm(full_address)
        
        if latitude is None or longitude is None:
            print("Failed to retrieve latitude and longitude. Exiting search.")
            return "No matching records found for the given Pincode or Address."
    
    distances, indices = tree.query([[latitude, longitude]], k=1)
    nearest_office = labels[indices[0][0]]
    nearest_pincode = pincodes[indices[0][0]]
    print(f"Nearest office predicted: {nearest_office} with Pincode: {nearest_pincode}")
    elapsed_time = time.time() - start_time
    return f"No exact match found. Nearest suggested office: {nearest_office} (Pincode: {nearest_pincode}) in {elapsed_time:.2f} seconds"


@app.route('/api/validate-address', methods=['POST'])
def validate_address():
    data = request.get_json()
    address = data.get('address')
    if not address:
        return jsonify({"error": "Address is required"}), 400
    
    result = search_office(address)
    return jsonify({"result": result}), 200

@app.route('/search', methods=['POST'])
def api_search_office():
    data = request.get_json()
    if 'address' not in data:
        return jsonify({"error": "Missing 'address' in request body"}), 400
    
    result = search_office(data['address'])
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
