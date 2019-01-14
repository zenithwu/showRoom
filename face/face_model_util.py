import math
import math
import os
import pickle
import time

import face_recognition
from face_recognition.face_detection_cli import image_files_in_folder
from sklearn import neighbors

def train_model(model_save_path=None,train_dir=None):
    n_neighbors=2
    knn_algo='ball_tree'
    verbose=False
    x = []
    y = []
    print("Training KNN classifier...")
    begin_time=time.time()
    for user_id in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, user_id)):
            continue
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, user_id)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                x.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(user_id)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(x))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(x, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    print("Training complete!")
    end_time=time.time()
    return str(end_time-begin_time)
if __name__ == '__main__':
    train_model(os.path.join("data/model","trained_knn_model.clf"),"data/train")


