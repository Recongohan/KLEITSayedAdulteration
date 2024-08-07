#Perfect
import sys
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, plot_roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

#REJECTION
from PIL import Image
def is_image_accepted(image_path):
    image = Image.open(image_path)
    pixels = image.load()
    width, height = image.size
    total_pixels = width * height
    red_sum, green_sum, blue_sum = 0, 0, 0

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            red_sum += r
            green_sum += g
            blue_sum += b

    average_red = red_sum // total_pixels
    average_green = green_sum // total_pixels
    average_blue = blue_sum // total_pixels

    if (110 <= average_red <= 255 and 25 <= average_green <= 92 and 5 <= average_blue <= 80):
        return True
    else:
        return False

# Usage example
imgpath=r"D:\Project KLE\Chilli Cropped\chilli1\test\10.jpg"
image_path = imgpath
if is_image_accepted(image_path):
    print("Image accepted!")
else:
    print("Image rejected!")
    sys.exit()
#REJECTION

# Define a function to extract color features from images
def color_features(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate the mean and standard deviation of hue, saturation and value channels
    mean_h = np.mean(hsv[:,:,0])
    mean_s = np.mean(hsv[:,:,1])
    mean_v = np.mean(hsv[:,:,2])
    std_h = np.std(hsv[:,:,0])
    std_s = np.std(hsv[:,:,1])
    std_v = np.std(hsv[:,:,2])
    # Return a feature vector of six values
    return [mean_h, mean_s, mean_v, std_h, std_s, std_v]

# Load the image data and labels
X = [] # Feature matrix
y = [] # Label vector
# Define a dictionary to map folder names to label values
label_map = {
    r'D:\Project KLE\Chilli Cropped\chilli1\pure\output': 0,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated5\output': 5,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated10\output': 10,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated15\output': 15,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated20\output': 20,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated25\output': 25,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated30\output': 30,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated35\output': 35,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated40\output': 40,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated45\output': 45,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated50\output': 50,
    r'D:\Project KLE\Chilli Cropped\chilli1\adulterated100\output':100
}

# Loop through the image folders
for folder in [r'D:\Project KLE\Chilli Cropped\chilli1\adulterated5\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated10\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated15\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated20\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated25\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated30\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated35\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated40\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated45\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated50\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\adulterated100\output',
           r'D:\Project KLE\Chilli Cropped\chilli1\pure\output']:
    # Get the path of the folder
    path = os.path.join(r"D:\Project KLE\Chilli Cropped\chilli1", folder)
    # Get the label for this folder
    label = label_map[folder]
    # Loop through the images in the folder 
    for file in os.listdir(path):
        # Get the full path of the image file
        file_path = os.path.join(path, file)
        # Read the image as a numpy array
        image = cv2.imread(file_path)
        # Extract color features from the image
        features = color_features(image)
        # Append the features to the feature matrix
        X.append(features)
        # Append the label to the label vector
        y.append(label)



# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a logistic regression model for binary classification
binary_model = LogisticRegression()
binary_model.fit(X[:2*len(os.listdir(path))], y[:2*len(os.listdir(path))])

# Create and train a logistic regression model for multiclass classification
multiclass_model = LogisticRegression()
multiclass_model.fit(X, y)

# Load the image to be predicted
test_image = cv2.imread(imgpath)

# Extract color features from the image
test_features = color_features(test_image)

# Convert the test features to numpy array
test_features = np.array(test_features).reshape(1,-1)

# Make a prediction using the model
prediction = multiclass_model.predict(test_features)
# Calculate the probability of the prediction
probability = multiclass_model.predict_proba(test_features)[0][1]


# Print the prediction and probability
if prediction == 0:
    print("The sample is classified as pure chilli powder")
    text = f"The sample is classified as pure chilli powder"
else:
    print("The sample is classified as adulterated chilli powder with {}% brick powder.".format(prediction))
    text = f"The sample is classified as adulterated chilli powder with {prediction}% brick powder."

cv2.putText(test_image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255, 0), 5)
cv2.namedWindow("Output Image", cv2.WINDOW_NORMAL)
cv2.imshow("Output Image", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Make predictions using the trained model
y_pred = multiclass_model.predict(X)

# Generate a confusion matrix
cm = confusion_matrix(y, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm, cmap='Blues', interpolation='nearest', vmin=0, vmax=100)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(label_map))
plt.xticks(tick_marks, label_map.keys(), rotation=45)
plt.yticks(tick_marks, label_map.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()

# Display the plot
plt.show()




# Make predictions on the test data
y_pred = multiclass_model.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the heatmap of the confusion matrix

sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Heat Map')
plt.show()


# Make predictions on the test set using the binary model
binary_preds = binary_model.predict(X_test)
binary_accuracy = accuracy_score(y_test, binary_preds)

# Make predictions on the test set using the multiclass model
multiclass_preds = multiclass_model.predict(X_test)
multiclass_accuracy = accuracy_score(y_test, multiclass_preds)

# Plot the accuracy scores on a graph
fig, ax = plt.subplots()
models = ['Binary Model', 'Multiclass Model']
accuracy_scores = [binary_accuracy, multiclass_accuracy]
sns.barplot(x=models, y=accuracy_scores, ax=ax)
ax.set_title('Accuracy Scores')
ax.set_ylabel('Accuracy')
plt.show()



# Use the trained models to predict labels for the test set
binary_preds = binary_model.predict(X_test)
multiclass_preds = multiclass_model.predict(X_test)

# Calculate the accuracy score for each model
binary_acc = accuracy_score(y_test, binary_preds)
multiclass_acc = accuracy_score(y_test, multiclass_preds)

# Create a bar chart to compare the accuracy scores of both models
plt.bar(['Binary Classification', 'Multiclass Classification'], [binary_acc, multiclass_acc])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim((0,1))
plt.show()
