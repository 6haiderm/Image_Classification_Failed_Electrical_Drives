# Electric Motor Production Quality Monitoring

## Project Overview
In the production of electrical drives, high product quality is essential to meet the demands of trends like **electric mobility** and continuing industrial automation. Traditional computer vision systems often reach their limits in flexible environments, such as **lot-size one production**. 

This project implements a **smart quality monitoring system** designed to detect assembly errors in electric motors using visual sensors. The system analyses images of motors from top and side perspectives to determine if a unit is fully assembled or contains specific defects.

## Features and Objectives
*   **Defect Detection:** Identifies three specific assembly defects: **missing cover**, **missing screw**, and **not screwed**.
*   **Multi-Model Analysis:** Investigates and compares the performance of **Support Vector Machines (SVM)**, **Artificial Neural Networks (ANN)**, and **Convolutional Neural Networks (CNN)**.
*   **Image Preprocessing:** Includes custom functions for resizing, color conversion (BGR to RGB or Grayscale), and 1D flattening for specific algorithms.
*   **Data Augmentation:** Implements techniques such as random flipping, rotation, and shifting to increase the dataset volume and improve model robustness.

## Dataset Description
The dataset consists of images captured from two perspectives:
1.  **Top View:** Used for initial analysis and primary defect detection.
2.  **Side View:** Helpful for detecting more subtle defects like "not screwed".

The data is distributed across four classes: **Complete**, **Missing cover**, **Missing screw**, and **Not screwed**. Note that the class distribution is imbalanced, with significantly fewer "Complete" assembly samples compared to defective ones, which poses a challenge for model training.

## Tech Stack
*   **Language:** Python
*   **Machine Learning/Deep Learning:** TensorFlow, Keras, Scikit-learn (SVM, OneHotEncoding, train_test_split)
*   **Computer Vision:** OpenCV (cv2)
*   **Data Manipulation:** NumPy, Pandas
*   **Visualisation:** Matplotlib, Seaborn

## Project Structure
The workflow is divided into two primary parts:
1.  **Step-by-Step Analysis:** Focuses on top-view images and a subset of defects (missing cover/screw) using SVMs and dense ANNs.
2.  **Advanced Detection:** Expands the system to detect "not screwed" defects using CNNs and side-perspective images, incorporating **data augmentation** for improved results.

## Methodology
### Data Preprocessing
Images are loaded and converted from the default OpenCV BGR format to **RGB**. For certain models like SVMs and fully connected ANNs, images are converted to grayscale and **flattened** into 1D vectors. CNNs, however, utilize the full multi-dimensional structure of the images.

### Model Architectures
*   **SVM:** Utilises an RBF kernel for classification.
*   **ANN:** A Sequential model consisting of multiple **Dense layers** with ReLU and Softmax activations.
*   **CNN:** Employs **Conv2D** layers, **MaxPooling2D**, and **GlobalMaxPooling2D** to excel at analyzing spatial image features.

### Evaluation
Models are evaluated using **confusion matrices** and **classification reports**, focusing on metrics such as precision, recall, and F1-score.

## Results Summary
Initial findings indicate that **CNNs perform better** than standard ANNs due to their ability to handle multi-dimensional image data with fewer parameters and better feature extraction through multiple layers. The application of **Data Augmentation** through the Keras `ImageDataGenerator` significantly improved model accuracy, reaching up to 86% in testing scenarios.

## References
This project draws background information from the paper: *Mayr et al., Machine Learning in Electric Motor Production - Potentials, Challenges and Exemplary Applications*.



##Image Classification: Using SVM and Deep learning( ANN, CNN) for detection of Faulty Electric drives

The target was to classify accurately the fully assembled electrical drives. Pictures of different sides of the drives were taken.
Models of Support Vector Machines and Convolutional Neural Networks were applied. Both models were analyzed and compared. Data augmentation was performed.

​**Google Collab environment was used and many Python libraries such as Scikitlearn, Keras, Tenser Flow were implemented in both projects. Further information is given in below links.

https://github.com/6haiderm/Image_Classification_Failed_Electrical_Drives/blob/main/Notebook/Exercise_Image_Classification.ipynb

##I used VS code, and Python to push the files to my GitHub Repository.
##Also, you can use the above-provided link to check the Google Collab Notebook.




