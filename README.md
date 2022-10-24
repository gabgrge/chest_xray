# Pneumonia_Detection
 Pneumonia detection machine learning model - Streamlit app

The code is organized in 5 functions :
 - load_data : Loads train, test and val images, configures them for use and returns them their associated label (pneumonia or not)
 - normalize : normalizes train, test and val sets
 - build_model : builds the deep learning classification model, compiles it, fits it for the train set then returns it with the accuracy
 - load_predict_image : if loaded, configures and displays image, then returns the prediction for this image (pneumonia or not)
 - main : configures web app, defines useful variables, connects and runs previous functions and displays results (image's prediction and accuracy)

Dataset source : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
