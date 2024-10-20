from flask import Flask, render_template, jsonify
import csv

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get-waypoints')
def get_waypoints():
    waypoints = []
    # Read the CSV file
    with open('waypoints_with_scores.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lat = float(row['Latitude'])
            lng = float(row['Longitude'])
            waypoints.append({'lat': lat, 'lng': lng})

    # Return the waypoints as JSON
    return jsonify(waypoints)

if __name__ == '__main__':
    app.run(debug=True)
