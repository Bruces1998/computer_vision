import numpy as np
import cv2
import os
class DatasetLoader:
        def __init__(self, preprocessors=None):
            self.preprocessors = preprocessors
            if self.preprocessors is None:
                self.preprocessors = []



        def load(self, imagePaths, verbose=-1):

            # initialize the list of features and labels
            data=[]
            labels=[]
            for (i, imagePath) in enumerate(imagePaths):
                # print(imagePath)
                # load the image and extract the class label assuming
                # that our path has the following format:
                # /path/to/dataset/{class}/{image}.jpg
                image = cv2.imread(imagePath)
                label = imagePath.split(os.path.sep)[-2]
                # data.append(image)
                # label.append(imagePath)
                # print(image, imagePath)                # label.append(imagePath)

                if self.preprocessors is not None:

                    for p in  self.preprocessors:
                        image = p.preprocess(image)


                data.append(image)
                labels.append(label)
                # print(label)
                # labels.append(label)
                # print(labels)

                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1,len(imagePaths)))


            return(np.array(data),np.array(labels))
