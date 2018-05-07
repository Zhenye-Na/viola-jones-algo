import numpy as np
import cv2
import os 
# face detection

# PCA algorithm
# pca theory http://blog.codinglabs.org/articles/pca-tutorial.html
class EigenFace(object): 
    def __init__(self,threshold,dimNum,dsize): 
        self.threshold = threshold  
        self.dimNum = dimNum 
        self.dsize = dsize 
  
    def loadImg(self,fileName,dsize): 
        ''''' 
        load image, rgb2gray, resize, equal histo 
        :param fileName: file name 
        :param dsize: same size, tuple 
        :return: image 
        '''
        img = cv2.imread(fileName)
        retImg = cv2.resize(img,dsize) 
        retImg = cv2.cvtColor(retImg,cv2.COLOR_BGR2GRAY) 
        retImg = cv2.equalizeHist(retImg)
        # cv2.imshow('img',retImg) 
        # cv2.waitKey() 
        return retImg 
  
  
    def createImgMat(self,dirName): 
        ''''' 
        generate training matrix 
        :param dirName: directory contains training data 
        :return: training data, training label 
        '''
        dataMat = np.zeros((10,1)) 
        label = [] 
        for parent,dirnames,filenames in os.walk(dirName): 
            # print parent 
            # print dirnames 
            # print filenames 
            index = 0
            for dirname in dirnames: 
                for subParent,subDirName,subFilenames in os.walk(parent+'/'+dirname): 
                    for filename in subFilenames: 
                        # print 'subParent',subParent
                        # print filename
                        if filename == '.DS_Store':
                            continue
                        img = self.loadImg(subParent+'/'+filename,self.dsize) 
                        tempImg = np.reshape(img,(-1,1)) 
                        if index == 0 : 
                            dataMat = tempImg 
                        else: 
                            dataMat = np.column_stack((dataMat,tempImg)) 
                        label.append(subParent+'/'+filename) 
                        index += 1
        return dataMat,label 
  
  
    def PCA(self,dataMat,dimNum): 
        ''''' 
        PCA to reduce the dimention 
        :param dataMat: training matrix 
        :param dimNum: dest dimention 
        :return: reduced training matrix and egen 
        '''
        #  
        meanMat = np.mat(np.mean(dataMat,1)).T 
        print 'mean matrix dimention:',meanMat.shape 
        diffMat = dataMat-meanMat 
        #  
        covMat = (diffMat.T*diffMat)/float(diffMat.shape[1]) # normalized 
        #covMat2 = np.cov(dataMat,bias=True) 
        #print 'cov by basic algor',covMat2 
        print 'cov',covMat.shape 
        eigVals, eigVects = np.linalg.eig(np.mat(covMat)) 
        print 'eig dimention',eigVects.shape 
        print 'eig value',eigVals 
        eigVects = diffMat*eigVects 
        eigValInd = np.argsort(eigVals) 
        eigValInd = eigValInd[::-1] 
        eigValInd = eigValInd[:dimNum] # chose the biggest n 
        print 'chosed eig value',eigValInd 
        eigVects = eigVects/np.linalg.norm(eigVects,axis=0) #normalize the eig vector 
        redEigVects = eigVects[:,eigValInd] 
        print 'chosed eig vector',redEigVects.shape 
        print 'mean matrix dimention',diffMat.shape 
        lowMat = redEigVects.T*diffMat 
        print 'low dimention matrix size:',lowMat.shape 
        return lowMat,redEigVects 
    
    def compare(self,dataMat,testImg,label): 
        ''''' 
        oru com. can also modifid to knn
        :param dataMat: train matrix 
        :param testImg: test image 
        :param label: train label 
        :return: file name that matched test 
        '''
        testImg = cv2.resize(testImg,self.dsize) 
        testImg = cv2.cvtColor(testImg,cv2.COLOR_RGB2GRAY) 
        testImg = np.reshape(testImg,(-1,1)) 
        lowMat,redVects = self.PCA(dataMat,self.dimNum) 
        testImg = redVects.T*testImg 
        print 'dim of test image after reshape',testImg.shape 
        disList = [] 
        testVec = np.reshape(testImg,(1,-1)) 
        for sample in lowMat.T: 
            disList.append(np.linalg.norm(testVec-sample)) 
        print disList 
        sortIndex = np.argsort(disList) 
        return label[sortIndex[0]]
  
    
    def predict(self,dirName,testFileName): 
        ''''' 
        pred function 
        :param dirName: dir of training data 
        :param testFileName: file name of test image 
        :return: pred result 
        '''
        testImg = cv2.imread(testFileName) 
        dataMat,label = self.createImgMat(dirName) 
        print 'loading image label',label 
        ans = self.compare(dataMat,testImg,label) 
        return ans 


# LBP algorithm

def loadnDetecFace(filename):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30))
    if (len(faces)==0):
        return None
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
            # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # cv2.imshow('img',roi_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return roi_gray

def trainingdata_prep(trainingImgTxt,trainLabelsTxt):
    with open(trainingImgTxt) as file:
        imgPath = file.readlines()
        imgPath = [s.rstrip() for s in imgPath]
    with open(trainLabelsTxt) as file:
        labels = file.readlines()
        labels = [s.rstrip() for s in labels]
    trainingFaces = []
    trainingLabels = []
    for i in range(len(imgPath)):
        print imgPath[i]
        face = loadnDetecFace(str(imgPath[i]))
        if face is not None:
            trainingFaces.append(face)
            trainingLabels.append(labels[i])
    return trainingFaces,trainingLabels

def face_recognizer_train(trainingImgTxt,trainLabelsTxt):
    trainingFaces,trainingLabels = trainingdata_prep(trainingImgTxt,trainLabelsTxt)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(trainingFaces,np.array(trainingLabels))
    return face_recognizer

def predict(face_recognizer,testImgtxt,testLabelsTxt):
    with open(testImgtxt) as file:
        imgPath = file.readlines()
    with open(testLabelsTxt) as file:
        labels = file.readlines()
    testindex = np.random.choice(100, 5, replace=False)
    testImgpath = []
    testLabels = []
    predLabels = []
    predconfidence = []
    for i in range(len(testindex)):
        testFace = loadnDetecFace(imgPath(i))
        if testFace is not None:
            testImgpath.append(imgPath(i))
            testLabels.append(labels(i))
            label, confidence = face_recognizer.predict(testFace)
            predLabels.append(label)
            predconfidence.append(confidence)
    return testImgpath,testLabels,predLabels,predconfidence


if __name__ == '__main__': 
    # eigenface = EigenFace(20,50,(50,50)) 
    # print eigenface.predict('./lfw','./test_face/Theodore_Tweed_Roosevelt_0003.jpg')
    # print eigenface.predict('./orl_faces','./test_face/10.pgm')
    # face = detecFace('./test_face/Theodore_Tweed_Roosevelt_0003.jpg')
    # face = loadnDetecFace('./lfw/German_Khan/German_Khan_0001.jpg')
    # cv2.imshow('img',face)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    face_recognizer = face_recognizer_train('trainingFilename.txt','traininglabel.txt')
    testImgpath,testLabels,predLabels,predconfidence = predict(face_recognizer,'testFilename.txt','testlabel.txt')
    print testImgpath,testLabels,predLabels,predconfidence
