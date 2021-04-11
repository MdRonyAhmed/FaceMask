import cv2
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


with_mask = np.load(r'With_Mask_500.npy')


im = Image.fromarray((with_mask[4]).reshape(50,50,3))
# im.show()

without_mask = np.load(r"Without_Mask_500.npy")

im = Image.fromarray((without_mask[5]).reshape(50,50,3))
# im.show()

with_mask = with_mask.reshape(501,1 * 50 * 50 * 3)
without_mask = without_mask.reshape(501,1 * 50 * 50 * 3)
X = np.r_[with_mask, without_mask]

labels = np.zeros(X.shape[0])

labels[501:] = 1.0
names = {0 : 'Mask', 1 : 'No Mask'}
color_dict={0:(0,255,0),1:(0,0,255)}

x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size = 0.35)
# print(x_train.shape)

pca = PCA(n_components = 3)
x_train = pca.fit_transform(x_train)

# print(x_train[0])

x_train.shape

x_train, x_test, y_train, y_test = train_test_split(X,labels, test_size = 0.25)

# print(x_train.shape)

svm = SVC()
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

print(accuracy_score(y_test, y_pred))

haar_data =cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    flag, img = capture.read()
    if flag:
        print('1')
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1, -1)

            pred = svm.predict(face)[0]
            n = names[int(pred)]
            color = color_dict[int(pred)]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
            cv2.putText(img, n, (x,y), font, 1, (255,250,250),2)

            if (int(pred)==1):
                cv2.putText(img, 'Please Wear Your Mask',(x-30, y+210),font, 0.6, (0,0,255),1)

        cv2.imshow('result', img)

        if cv2.waitKey(2) == 27:
            break


capture.release()
cv2.destroyAllWindows()
