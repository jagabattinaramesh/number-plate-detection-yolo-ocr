Collect the data from:
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/data
https://universe.roboflow.com/augmented-startups/vehicle-registration-plates-trudk
https://universe.roboflow.com/recognision-datasets/uk-number-plate-recognision.

Merge all the data and given proper id’s to each image using python script.

Divided the data into train(7009-images), validation(1864-images) and test(928-images). 

Data sets have different annotation formats like Pascal VOC and COCO. Converted all to YOLO annotation format.

Use CVAT for giving annotations and also to check annotations after merging from different sources.



Next create a virtual machine in google cloud with t4 gpu for training yolo model on our data.

Clone Ultralytics git repository: “https://github.com/ultralytics/ultralytics.git”

Train the yolo model on my data using ultralytics git repository

Store the best.pt which bives best weights

Next use python code to crop the images to locate exact number plate using bounding boxes from yolo.

Use PaddleOcr to detect the alphanumerics from cropped number plate

Store the results in a json file.

Use FastAPI to deploy the model

