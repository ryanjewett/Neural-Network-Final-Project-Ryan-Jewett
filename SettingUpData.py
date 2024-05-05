import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

ROWS = 64
COLS = 64
CHANNELS = 1

def read_image(file_path):
  img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (ROWS, COLS),interpolation=cv2.INTER_CUBIC)
  return img

def PrepData(images):
    m = len(images)
    n_x = ROWS * COLS * CHANNELS

    X = np.ndarray((n_x, m), dtype=np.uint8)
    y = np.zeros((m, 1))
    print("X.shape is {}".format(X.shape))

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        X[:, i] = np.squeeze(image.reshape((n_x, 1)))

        if image_file.startswith("imagesallfiles/Zero_full"):
            label = 0
        elif image_file.startswith("imagesallfiles/One_full"):
            label = 1
        elif image_file.startswith("imagesallfiles/Two_full"):
            label = 2
        elif image_file.startswith("imagesallfiles/Three_full"):
            label = 3
        elif image_file.startswith("imagesallfiles/Four_full"):
            label = 4
        elif image_file.startswith("imagesallfiles/Five_full"):
            label = 5
        elif image_file.startswith("imagesallfiles/Six_full"):
            label = 6
        elif image_file.startswith("imagesallfiles/Seven_full"):
            label = 7
        elif image_file.startswith("imagesallfiles/Eight_full"):
            label = 8
        elif image_file.startswith("imagesallfiles/Nine_full"):
            label = 9
        else:
            # Handle the case when the file name doesn't match expected patterns
            label = 0  # or any other appropriate default value
        print(f"Set label to {label}")
        y[i,0] = label
        print(f"Image {i} done!")
    return X, y

def main():
  
    trainingDataFolder ='imagesfiles'
    train_images = [os.path.join(trainingDataFolder, i) for i in os.listdir(trainingDataFolder)]
    
    finalImages,labledArray = PrepData(train_images)
    finalImages = finalImages.T
    finalImages = finalImages.reshape(-1,ROWS,COLS,CHANNELS)
    print("Final Images Shape:", finalImages.shape)
    print("Labeled Array:", labledArray.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(finalImages, labledArray, test_size=0.2, random_state=42)

    np.savez_compressed('imageData3', X_train, X_test, Y_train, Y_test)
    print("done")


if __name__ == "__main__":
    main()
