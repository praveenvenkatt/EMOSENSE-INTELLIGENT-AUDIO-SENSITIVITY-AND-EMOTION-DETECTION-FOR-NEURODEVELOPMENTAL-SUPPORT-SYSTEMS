## Title of the Project
Intelligent Audio Sensitivity and Emotion Detection for Neurodevelopmental Support Systems

An AI-driven assistive system that integrates real-time facial emotion detection and audio sensitivity analysis for neurodevelopmental support.

## About

EmoSense is an AI-driven web-based assistive learning system designed to support children with neurodevelopmental conditions, primarily Autism Spectrum Disorder (ASD). The project focuses on improving emotional understanding and sensory awareness through real-time facial emotion recognition and intelligent audio sensitivity analysis.

Traditional emotional learning methods rely on manual observation and static learning tools, which lack real-time interaction and measurable feedback. EmoSense addresses these limitations by using deep learning–based emotion detection, emoji-based visual learning, structured response tracking, and automated progress reporting for parents and caregivers.

## Features

- Real-time facial emotion recognition using InceptionV3 (CNN)
- Interactive emoji-based emotional learning interface
- Multi-attempt response tracking with automated evaluation
- Intelligent audio sensitivity detection through facial reaction monitoring
- Automated email-based progress report generation
- Scalable and web-based deployment architecture

## Requirements
<!--List the requirements of the project as shown below-->
* Operating System: Requires a 64-bit OS (Windows 10) for compatibility with deep learning frameworks.
* Development Environment: Python 3.10 or later is necessary for coding the sign language detection system.
* Deep Learning Frameworks: TensorFlow for model training
* Image Processing Libraries: OpenCV is essential for efficient image processing and real-time hand gesture recognition.
* Version Control: Implementation of Git for collaborative development and effective code management.
* IDE: Use of VSCode as the Integrated Development Environment for coding, debugging, and version control integration.
* Additional Dependencies: Includes scikit-learn, TensorFlow (versions 2.4.1), TensorFlow GPU, OpenCV for deep learning tasks.

## System Architecture

<img width="1322" height="750" alt="{2A4B1A8E-809D-47A0-A1AF-B4C81EDDDF2B}" src="https://github.com/user-attachments/assets/8b75aedc-faa2-4e16-81e1-e09e3c42f12a" />

## Project coding part
### Autism_Training
```
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, 
Flatten,GlobalAveragePooling2D 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import ReduceLROnPlateau 
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten 
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.inception_v3 import InceptionV3 
from tensorflow.keras.applications.inception_v3 import preprocess_input 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img 
from tensorflow.keras.models import Sequential 
import numpy as np 
from glob import glob 
train_datagen = ImageDataGenerator(rescale=1./255, 
shear_range=0.2, 
zoom_range=0.2, 
horizontal_flip=True) 
train_datagen = ImageDataGenerator(rescale=1./255) 
#Training & Testing Spliting 
training_set = train_datagen.flow_from_directory(r'dataset1/train', 
target_size=(224, 224), 
batch_size=32, 
class_mode='categorical') 
30 
test_set = train_datagen.flow_from_directory(r'dataset1/validation', 
target_size=(224, 224), 
batch_size=32, 
class_mode='categorical') 
inception_v3 = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, 
input_shape=(224, 224, 3)) 
for layer in inception_v3.layers[: -15]: 
layer.trainable = False 
x = inception_v3.output 
x = Flatten()(x) 
x = Dense(units=512, activation='relu')(x) 
x = Dropout(0.3)(x) 
x = Dense(units=512, activation='relu')(x) 
x = Dropout(0.3)(x) 
output = Dense(units=7, activation='softmax')(x) 
model = Model(inception_v3.input, output) 
#model.summary() 
loss = tf.keras.losses.CategoricalCrossentropy() 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) 
histry = model.fit(training_set, 
validation_data=test_set, 
epochs=100, 
steps_per_epoch=len(training_set), 
validation_steps=len(test_set), 
) 
plt.plot(histry.history['loss'], label='train loss') 
plt.legend() 
plt.show() 
# plot the accuracy 
plt.plot(histry.history['accuracy'], label='accuracy') 
31 
plt.legend() 
plt.show() 
model.save('autism_model.h5')
```

### Training_Model
```
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, 
Flatten,GlobalAveragePooling2D 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import ReduceLROnPlateau 
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten 
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.inception_v3 import InceptionV3 
from tensorflow.keras.applications.inception_v3 import preprocess_input 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img 
from tensorflow.keras.models import Sequential 
import numpy as np 
from glob import glob 
train_datagen = ImageDataGenerator(rescale=1./255, 
shear_range=0.2, 
zoom_range=0.2, 
horizontal_flip=True) 
train_datagen = ImageDataGenerator(rescale=1./255) 
#Training & Testing Spliting 
training_set = train_datagen.flow_from_directory(r'dataset/train', 
target_size=(224, 224), 
batch_size=32, 
32 
class_mode='categorical') 
test_set = train_datagen.flow_from_directory(r'dataset/validation', 
target_size=(224, 224), 
batch_size=32, 
class_mode='categorical') 
inception_v3 = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, 
input_shape=(224, 224, 3)) 
for layer in inception_v3.layers[: -15]: 
layer.trainable = False 
x = inception_v3.output 
x = Flatten()(x) 
x = Dense(units=512, activation='relu')(x) 
x = Dropout(0.3)(x) 
x = Dense(units=512, activation='relu')(x) 
x = Dropout(0.3)(x) 
output = Dense(units=7, activation='softmax')(x) 
model = Model(inception_v3.input, output) 
#model.summary() 
loss = tf.keras.losses.CategoricalCrossentropy() 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) 
histry = model.fit(training_set, 
validation_data=test_set, 
epochs=100, 
steps_per_epoch=len(training_set), 
validation_steps=len(test_set), 
) 
plt.plot(histry.history['loss'], label='train loss') 
plt.legend() 
plt.show() 
# plot the accuracy 
33 
plt.plot(histry.history['accuracy'], label='accuracy') 
plt.legend() 
plt.show() 
model.save('model1.h5') 
import numpy as np 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
import seaborn as sns 
import matplotlib.pyplot as plt 
# Get true labels and predictions 
y_true = test_set.classes  # Actual labels 
y_pred = np.argmax(model.predict(test_set), axis=1)  # Predicted labels 
# Get class names 
class_labels = list(test_set.class_indices.keys()) 
# Compute confusion matrix 
conf_matrix = confusion_matrix(y_true, y_pred) 
# Plot confusion matrix 
plt.figure(figsize=(8, 6)) 
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, 
yticklabels=class_labels) 
plt.xlabel("Predicted Label") 
plt.ylabel("True Label") 
plt.title("Confusion Matrix") 
plt.show() 
# Print classification report (Precision, Recall, F1-score) 
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_labels)) 
# Print overall accuracy 
accuracy = accuracy_score(y_true, y_pred) 
print("Accuracy: {:.2f}%".format(accuracy * 100)) 
```
### App.py
```
from flask import Flask, render_template, request, redirect, url_for, session,Response 
import cv2 
34 
import numpy as np 
import tensorflow as tf 
import time   
import smtplib 
from email.mime.text import MIMEText 
from email.mime.multipart import MIMEMultipart 
from email.mime.base import MIMEBase 
from email import encoders 
from keras.models import load_model 
import os 
import ssl 
from flask import Flask, Response, jsonify 
from collections import Counter 
app = Flask(__name__) 
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W' 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
model1 = tf.keras.models.load_model('autism_model.h5') 
class_label = ['anger', 'fear', 'joy', 'natural', 'sadness', 'surprise'] 
email_sent = False 
emotion_history = [] 
emotion_capture_threshold = 10   
frame_counter = 0 
time_window = 100 
dominant_emotion = "" 
latest_emotion = dominant_emotion 
model = tf.keras.models.load_model('model1.h5') 
UPLOAD_FOLDER = 'static/uploads' 
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
@app.route('/', methods=['GET', 'POST']) 
def index(): 
if request.method == 'POST': 
35 
total_score = sum(int(request.form.get(f'q{i}', 0)) for i in range(1, 10)) 
result = detect_depression(total_score) 
return render_template('result.html', score=total_score, result=result) 
return render_template('index.html') 
@app.route('/camera') 
def camera(): 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
'haarcascade_frontalface_default.xml') 
cap = cv2.VideoCapture(0) 
class_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise'] 
def preprocess_frame(face): 
face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)   
face = cv2.resize(face, (224, 224))   
face = np.expand_dims(face, axis=-1)   
face = np.repeat(face, 3, axis=-1)   
face = face / 255.0   
face = np.expand_dims(face, axis=0)   
return face 
def predict_emotion(face): 
img_array = preprocess_frame(face) 
prediction = model.predict(img_array)   
predicted_class_index = np.argmax(prediction, axis=1)[0]   
predicted_class_label = class_labels[predicted_class_index] 
confidence = np.max(prediction)   
return predicted_class_label, confidence 
detected_emotions = [] 
start_time = time.time()   
36 
while cap.isOpened(): 
ret, frame = cap.read() 
if not ret: 
break 
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
gray_frame_3channel = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR) 
faces = face_cascade.detectMultiScale(gray_frame_3channel, scaleFactor=1.1, minNeighbors=5, 
minSize=(30, 30)) 
for (x, y, w, h) in faces: 
face = frame[y:y + h, x:x + w] 
emotion, confidence = predict_emotion(face) 
detected_emotions.append((emotion, confidence)) 
label = f"{emotion} ({confidence*100:.2f}%)" 
cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 
cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
cv2.imshow('Emotion Detection', frame) 
if time.time() - start_time > 120:   
break   
if cv2.waitKey(1) & 0xFF == ord('q'): 
break 
cap.release() 
cv2.destroyAllWindows() 
37 
if detected_emotions: 
emotions, confidences = zip(*detected_emotions) 
most_common_emotion = max(set(emotions), key=emotions.count) 
avg_confidence = sum(confidences) / len(confidences) 
else: 
most_common_emotion = "No face detected" 
avg_confidence = 0.0 
session['emotion'] = most_common_emotion 
session['confidence'] = round(avg_confidence * 100, 2) 
return redirect(url_for('emotion_result')) 
emotion_emoji_map = { 
"anger": "        
", 
"disgust": "        
"fear": "         
"happiness": "       
"neutrality": "      
"sadness": "        
"surprise": "         
} 
", 
", 
", 
", 
", 
" 
@app.route('/emotion_result') 
def emotion_result(): 
emotion = session.get('emotion', "Unknown") 
emoji = emotion_emoji_map.get(emotion, "  
")   
return render_template('emotion_result.html', emotion=emotion.capitalize(), emoji=emoji) 
def send_feedback_email(subject, feedback): 
sender_email = "sraghul154@gmail.com" 
receiver_email = "praveenofficial2207@gmail.com" 
password = "nrwdkqkivzywvhzt" 
38 
msg = MIMEMultipart() 
msg["Subject"] = subject 
msg["From"] = sender_email 
msg["To"] = receiver_email 
msg.attach(MIMEText(feedback, "plain"))   
try: 
server = smtplib.SMTP("smtp.gmail.com", 587) 
server.starttls() 
server.login(sender_email, password) 
server.sendmail(sender_email, receiver_email, msg.as_string()) 
server.quit() 
print("Feedback email sent successfully!") 
return True 
except Exception as e: 
print("Error sending feedback email:", e) 
return False 
@app.route('/autism') 
def autism(): 
session['attempts'] = 0 
session['correct'] = 0 
return render_template('autism.html') 
@app.route("/submit", methods=["POST"]) 
def submit_feedback(): 
data = request.get_json() 
clicked_emotion = data.get("emotion") 
detected_emotion = session.get('emotion', "Unknown") 
if "attempts" not in session: 
session['attempts'] = 0 
session['correct'] = 0 
39 
session['attempts'] += 1 
if clicked_emotion == detected_emotion: 
session['correct'] += 1 
if session['attempts'] == 3: 
subject = "" 
feedback = "" 
if session['correct'] == 3: 
subject = "Learning Well" 
feedback = "User correctly identified the emotion 3 times." 
elif session['correct'] == 2: 
subject = "Learning Much Better" 
feedback = "User correctly identified the emotion 2 times." 
elif session['correct'] == 0: 
subject = "Learning Not Well" 
feedback = "User incorrectly identified the emotion 3 times." 
send_feedback_email(subject, feedback)   
session['attempts'] = 0   
session['correct'] = 0 
return jsonify({"message": "Feedback received."}) 
def preprocess_frame(frame): 
"""Preprocess the face before feeding it to the model.""" 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
if len(faces) == 0: 
return None, None 
for (x, y, w, h) in faces: 
face = frame[y:y + h, x:x + w] 
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) 
40 
face = cv2.resize(face, (224, 224)) 
face = face.astype('float32') / 255.0 
face = np.expand_dims(face, axis=0) 
return face, (x, y, w, h) 
return None, None 
def predict_emotion(frame): 
"""Predict the emotion from the detected face.""" 
processed_face, face_coords = preprocess_frame(frame) 
if processed_face is not None and model1 is not None: 
prediction = model1.predict(processed_face)[0] 
emotion_index = np.argmax(prediction) 
emotion = class_label[emotion_index] 
confidence = float(prediction[emotion_index]) 
return emotion, confidence, face_coords 
return "", 0.0, None   
def send_email(image_path, detected_emotion): 
"""Send an email with the captured emotion image and detected emotion text.""" 
sender_email = "sraghul154@gmail.com" 
receiver_email = "praveenofficial2207@gmail.com" 
password = "nrwdkqkivzywvhzt" 
subject = f"Emotion Detected: {detected_emotion}" 
body = f"The detected dominant emotion is: {detected_emotion}\n\nAttached is the captured 
frame." 
msg = MIMEMultipart() 
msg['From'] = sender_email 
msg['To'] = receiver_email 
msg['Subject'] = subject 
msg.attach(MIMEText(body, "plain")) 
with open(image_path, "rb") as attachment: 
part = MIMEBase("application", "octet-stream") 
part.set_payload(attachment.read()) 
encoders.encode_base64(part) 
41 
part.add_header("Content-Disposition", f"attachment; filename=emotion.jpg") 
msg.attach(part) 
context = ssl.create_default_context() 
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server: 
server.login(sender_email, password) 
server.sendmail(sender_email, receiver_email, msg.as_string()) 
global email_sent 
email_sent = True   
tracked_faces = {}   
def generate_frames(): 
"""Capture video frames, analyze emotions, and send email for the most dominant emotion.""" 
global email_sent, frame_counter 
cap = cv2.VideoCapture(0) 
while True: 
success, frame = cap.read() 
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, 
minSize=(30, 30)) 
for (x, y, w, h) in faces: 
emotion, confidence, face_coords = predict_emotion(frame) 
if emotion: 
emotion_history.append(emotion) 
label = f"{emotion} ({confidence*100:.2f}%)" 
cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 
cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
frame_counter += 1 
if frame_counter >= time_window and not email_sent: 
if emotion_history: 
dominant_emotion = Counter(emotion_history).most_common(1)[0][0]   
image_path = "dominant_emotion.jpg" 
cv2.imwrite(image_path, frame) 
42 
send_email(image_path, dominant_emotion) 
email_sent = True   
emotion_history.clear() 
frame_counter = 0 
email_sent = False   
_, buffer = cv2.imencode('.jpg', frame) 
frame_bytes = buffer.tobytes() 
yield (b'--frame\r\n' 
b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') 
cap.release() 
@app.route('/video_feed') 
def video_feed(): 
"""Stream the live video feed.""" 
return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame') 
@app.route('/emotion_data') 
def emotion_data(): 
"""Return latest detected emotion data as JSON.""" 
return jsonify(latest_emotion) 
@app.route('/audio', methods=['GET']) 
def audio(): 
return render_template('audio.html') 
@app.route('/upload', methods=['POST']) 
def upload(): 
"""Handle audio file upload and save it.""" 
if 'audio' not in request.files: 
return jsonify({"error": "No file uploaded"}), 400 
audio_file = request.files['audio'] 
audio_path = "static/uploads/" + audio_file.filename 
audio_file.save(audio_path) 
return jsonify({"audio_url": f"/{audio_path}"}) 
if __name__ == '__main__': 
app
```



## Output
#### Output1 - Home Page

<img width="1189" height="728" alt="{F5E4CB6B-B1D3-40E2-BC1C-E59C926133DD}" src="https://github.com/user-attachments/assets/a06078c0-57e5-4f52-b541-6be8b6efadf1" />


#### Output2 - Emotion Detection Output
<img width="981" height="536" alt="{E42B5CF7-DB96-4DF2-A7A8-F3D245D18450}" src="https://github.com/user-attachments/assets/4f6d7955-16df-47da-badc-fbc64e5c3fe0" />


#### Output3 - Emotion Selection Interface
<img width="985" height="548" alt="{8669E33B-9377-4C94-A78D-0C4E673133A4}" src="https://github.com/user-attachments/assets/bcbe4663-c6c8-42c8-8bdb-ddf8e8fb3e58" />

## Model Performance
Training Accuracy: 95-97%.
(Based on FER2013 dataset and InceptionV3 transfer learning)



## Results and Impact

The system improves emotional understanding, sensory awareness, and parent–child interaction using AI-driven analysis. EmoSense supports inclusive learning and provides data-driven insights for caregivers and educators in neurodevelopmental support.

## Articles published / References
[1]    M. A. Rashidan et al., "Technology-Assisted Emotion Recognition for Autism Spectrum Disorder 
(ASD) Children: A Systematic Literature Review," in IEEE Access, vol. 9, pp. 33638-33653, 2021, 
doi: 10.1109/ACCESS.2021.3060753. 

[2]    Y. -L. Chien et al., "Game-Based Social Interaction Platform for Cognitive Assessment of Autism 
Using Eye Tracking," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 
31, pp. 749-758, 2023, doi: 10.1109/TNSRE.2022.3232369.  

[3]    K. D. Bartl-Pokorny et al., "Robot-Based Intervention for Children With Autism Spectrum 
Disorder: A Systematic Literature Review," in IEEE Access, vol. 9, pp. 165433-165450, 2021, doi: 
10.1109/ACCESS.2021.3132785. 

[4]    V. G. Prakash et al., "Computer Vision-Based Assessment of Autistic Children: Analyzing 
Interactions, Emotions, Human Pose, and Life Skills," in IEEE Access, vol. 11, pp. 47907-47929, 2023, 
doi: 10.1109/ACCESS.2023.3269027. 

[5]     A. Kurian and S. Tripathi, "m_AutNet–A Framework for Personalized Multimodal Emotion 
Recognition in Autistic Children," in IEEE Access, vol. 13, pp. 1651-1662, 2025, doi: 
10.1109/ACCESS.2024.3403087.   

[6]   H. Dong, D. Chen, L. Zhang, H. Ke, and X. J. Li, “Subject sensitive EEGdiscrimination with fast 
reconstructable CNN driven by reinforcementlearning: A case study of ASD evaluation,” 
Neurocomputing, vol. 449,pp. 136–145, Aug. 2021, doi: 10.1016/j.neucom.2021.04.009. 

[7]    M. Ranjani and P. Supraja, “Classifying the autism and epilepsy disorderbased on EEG signal 
using deep convolutional neural network (DCNN),”in Proc. Int. Conf. Advance Comput. Innov. 
Technol. Eng. (ICACITE),Mar. 2021, pp. 880–886




