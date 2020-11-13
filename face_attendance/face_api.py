from flask import Flask, jsonify, request
import pickle
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import time
from io import BytesIO
import base64
import json

# Initialize the Flask application
app = Flask(__name__)

# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['jpg','jpeg','png'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api/facedetect', methods=['GET','POST'])
def upload_image():
    if request.method == "GET":
        resp = jsonify({"message":"Error", "data'":"Method not allowed"})
        resp.status_code = 405
        return resp
    else:
        # check if the post request has the files part
        requestData = request.form
        if 'face' not in requestData:
            resp = jsonify({"message":"Error", "data'":"No face found"})
            resp.status_code = 500
            return resp
        # Get the list of the uploaded files
        file = BytesIO(base64.decodebytes(requestData["face"].encode()))
        start = time.time()
        results = []
        with open("trained_knn_model.clf", 'rb') as f:
            knn_clf = pickle.load(f)
            
            image = face_recognition.load_image_file(file)
            X_face_locations = face_recognition.face_locations(image)
            
            if len(X_face_locations) != 0:
                # Find encodings for faces in the test iamge
                faces_encodings = face_recognition.face_encodings(image, known_face_locations=X_face_locations)

                # Use the KNN model to find the best matches for the test face
                closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
                are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(X_face_locations))]
                predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
                lp = 0
                for name, (top, right, bottom, left) in predictions:
                    resarray = {}
                    resarray["name"] = name
                    resarray["accuracy"] = closest_distances[0][lp][0]
                    results.append(resarray)
                    lp = lp + 1

        print(results)
        if(results is None):
            resp = jsonify({"message":"Error", "data'":"No face found"})
            resp.status_code = 500
            return resp
        else:
            resp = jsonify({"message":"success", "data": results})
            resp.status_code = 200
            return resp

if __name__ == '__main__':
    app.run(port=5000, host="192.168.43.91", debug=True, threaded=False, processes=3)
