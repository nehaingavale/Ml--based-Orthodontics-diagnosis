from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from flask_ngrok import run_with_ngrok 
from PIL import Image
import os
import sys
import cv2
from test_images import *



app = Flask(__name__)
run_with_ngrok(app)
UPLOAD_FOLDER = '/content/drive/MyDrive/ObjectDetection_FlaskDeployment-master/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
#app.config['DETECTION_FOLDER'] = DETECTION_FOLDER

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/about")
def about():
  return render_template("about.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      # create a secure filename
      filename = secure_filename(f.filename)
      print(filename)
      # save file to /static/uploads
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      print(filepath)
      f.save(filepath)
      get_image(filepath, filename)
      
      return render_template("uploaded.html", display_detection = filename, fname = filename)      

if __name__ == '__main__':
   app.run()
