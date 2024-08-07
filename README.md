# KLEITSayedAdulteration
# Chilli Powder Adulteration Detection

## Project Description

This project aims to detect adulteration in chilli powder by analyzing images of the powder. Using machine learning techniques, specifically Logistic Regression, the project classifies chilli powder samples into pure or various levels of adulteration based on color features extracted from the images.

## Usefulness

Adulteration in food products, especially spices like chilli powder, is a significant concern for consumers and regulatory bodies. Adulteration can affect the quality, taste, and safety of the food. This project provides a tool to automatically and accurately detect adulteration in chilli powder, which can be useful for:

- **Consumers**: Ensuring the purity of the chilli powder they purchase.
- **Regulatory Bodies**: Monitoring and controlling the quality of spices in the market.
- **Food Manufacturers**: Maintaining the quality and safety of their products.

## How It Works

### Feature Extraction

The project extracts color features from images of chilli powder. These features include the mean and standard deviation of the hue, saturation, and value (HSV) channels of the images.

### Data Preparation

Images are organized into folders representing different levels of adulteration. These images are then processed to extract color features, which are used to create a feature matrix `X` and a label vector `y`.

### Model Training

Two Logistic Regression models are trained:

1. **Binary Classification Model**: This model differentiates between pure and adulterated chilli powder.
2. **Multiclass Classification Model**: This model classifies the level of adulteration in the chilli powder.

### Making Predictions

The trained models are used to predict the class of new images. The prediction indicates whether the sample is pure or the level of adulteration present.

### Model Evaluation

The performance of the models is evaluated using confusion matrices and accuracy scores. Visualizations such as heatmaps and bar charts are used to present the evaluation results.

## Example Usage

1. **Load Image**: An image of the chilli powder is loaded.
2. **Extract Features**: Color features are extracted from the image.
3. **Predict Class**: The trained model predicts whether the chilli powder is pure or adulterated and, if adulterated, the level of adulteration.
4. **Display Results**: The prediction results are displayed on the image, and performance metrics are visualized.

## Conclusion

This project demonstrates the application of machine learning in detecting food adulteration using image analysis. By leveraging Logistic Regression and color feature extraction, the project provides a practical solution for ensuring the quality and safety of chilli powder.
