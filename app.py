from flask import Flask, render_template, request, redirect, url_for
import requests
app = Flask(__name__)
from flask import Flask
from flask_pymongo import PyMongo
# from another import home_bp
from bson.binary import Binary
import base64
from bson.objectid import ObjectId
import tensorflow as tf
from tensorflow.keras import layers,models
from PIL import Image
import numpy as np
import io




app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
mongo = PyMongo(app)
#    mongo.db.inventory.insert_one({"a":name,"email":email})
# -------------------------ML--------------------------------
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # Change 10 to the number of classes you have
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -------------------------ML End------------------------------



def preprocess_image(image_data):
    # Convert binary data to a PIL image
    image = Image.open(io.BytesIO(image_data))
    
    # Resize the image to the expected input size of the model
    image = image.resize((224, 224))  # Change dimensions according to your model
    image = np.array(image) / 255.0    # Normalize the image data
    
    # Expand dimensions to match the model input shape
    image = np.expand_dims(image, axis=0)
    return image

 
@app.route('/')
def home():
    

    api_url = "https://jsonplaceholder.typicode.com/todos/"
    response = requests.get(api_url)
    

    if response.status_code == 200:
        todos = response.json()  
    else:
        todos = [] 

    return render_template('home.html', todos=todos)


 
# About page
@app.route('/about',methods=['GET'])

def about():
 
    users_list=mongo.db.users.find()
    return render_template('about.html',users=users_list)



# Services page with some dynamic content
@app.route('/services',methods=['GET','POST'])

def services():
    latest_image = None  # Initialize variable for the latest image

    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                # Read the image file as binary data
                image_data = image.read()

                # Store the image in MongoDB
                mongo.db.images.insert_one({
                    'image_name': image.filename,
                    'image_data': Binary(image_data)  # Save image as binary data
                })
                preprocessed_image=preprocess_image(image_data)
                predictions=model.predict(preprocessed_image)
                return redirect(url_for('services'))  # Redirect to the same page to display the latest image

    # Retrieve the most recently uploaded image from MongoDB
    latest_image_doc = mongo.db.images.find().sort('_id', -1).limit(1)
    for image in latest_image_doc:
        # Encode the image data to base64
        image['image_data'] = base64.b64encode(image['image_data']).decode('utf-8')
        latest_image = image  # Get the latest image

    return render_template('services.html', latest_image=latest_image)


# Contact page with form
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Save the information to MongoDB
        mongo.db.users.insert_one({'name': name, 'email': email, 'message': message})
        mongo.db.inventory.insert_one({"a":name,"email":email})

        # Redirect to a success page or back to the form
        return redirect(url_for('success'))

    return render_template('contact.html')

@app.route('/success')
def success():
    return "Your message has been sent successfully!"


@app.route('/analysis')
def images():
    images_list = mongo.db.images.find()
    for image in images_list:
        image['image_data'] = base64.b64encode(image['image_data']).decode('utf-8')
    
    return render_template('analysis.html', images=images_list)

#----------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
