##Image Classification: Using SVM and Deep learning( ANN, CNN) for detection of Faulty Electric drives

The target was to classify accurately the fully assembled electrical drives. Pictures of different sides of the drives were taken.
Models of Support Vector Machines and Convolutional Neural Networks were applied. Both models were analyzed and compared. Data augmentation was performed.

​**Google Collab environment was used and many Python libraries such as Scikitlearn, Keras, Tenser Flow were implemented in both projects. Further information is given in below links.

https://github.com/6haiderm/Image_Classification_Failed_Electrical_Drives/blob/main/Notebook/Exercise_Image_Classification.ipynb

##I used VS code, and Python to push the files to my GitHub Repository.
##Also, you can use the above-provided link to check the Google Collab Notebook.



flowchart TB
    CAM[Image Acquisition<br/>(Industrial Cameras)]
    PRE[Image Preprocessing<br/>• Resize<br/>• Normalize<br/>• Denoise]
    FEAT[Feature Representation]
    SVM[SVM Classifier]
    CNN[CNN Classifier]
    OUT[Assembly State Prediction<br/>(4 Classes)]
    QC[Quality Control Decision]

    CAM --> PRE
    PRE --> FEAT
    FEAT --> SVM
    FEAT --> CNN
    SVM --> OUT
    CNN --> OUT
    OUT --> QC
