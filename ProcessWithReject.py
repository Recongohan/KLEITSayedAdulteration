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

imgpath = r"D:\Project KLE\Chilli Cropped\chilli1\test\10.jpg"
image_path = imgpath
if is_image_accepted(image_path):
    print("Image accepted!")
else:
    print("Image rejected!")
    sys.exit()

def color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:,:,0])
    mean_s = np.mean(hsv[:,:,1])
    mean_v = np.mean(hsv[:,:,2])
    std_h = np.std(hsv[:,:,0])
    std_s = np.std(hsv[:,:,1])
    std_v = np.std(hsv[:,:,2])
    return [mean_h, mean_s, mean_v, std_h, std_s, std_v]

X = []
y = []
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
    path = os.path.join(r"D:\Project KLE\Chilli Cropped\chilli1", folder)
    label = label_map[folder]
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        image = cv2.imread(file_path)
        features = color_features(image)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

binary_model = LogisticRegression()
binary_model.fit(X[:2*len(os.listdir(path))], y[:2*len(os.listdir(path))])

multiclass_model = LogisticRegression()
multiclass_model.fit(X, y)

test_image = cv2.imread(imgpath)
test_features = color_features(test_image)
test_features = np.array(test_features).reshape(1,-1)

prediction = multiclass_model.predict(test_features)
probability = multiclass_model.predict_proba(test_features)[0][1]

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

y_pred = multiclass_model.predict(X)
cm = confusion_matrix(y, y_pred)

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
plt.show()

y_pred = multiclass_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Heat Map')
plt.show()

binary_preds = binary_model.predict(X_test)
binary_accuracy = accuracy_score(y_test, binary_preds)

multiclass_preds = multiclass_model.predict(X_test)
multiclass_accuracy = accuracy_score(y_test, multiclass_preds)

fig, ax = plt.subplots()
models = ['Binary Model', 'Multiclass Model']
accuracy_scores = [binary_accuracy, multiclass_accuracy]
sns.barplot(x=models, y=accuracy_scores, ax=ax)
ax.set_title('Accuracy Scores')
ax.set_ylabel('Accuracy')
plt.show()

binary_preds = binary_model.predict(X_test)
multiclass_preds = multiclass_model.predict(X_test)

binary_acc = accuracy_score(y_test, binary_preds)
multiclass_acc = accuracy_score(y_test, multiclass_preds)

plt.bar(['Binary Classification', 'Multiclass Classification'], [binary_acc, multiclass_acc])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim((0,1))
plt.show()
