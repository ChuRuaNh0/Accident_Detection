from create_data import make_dataset
import numpy as np 
from keras.models import load_model
import tensorflow as tf
import cv2
from PIL import Image, ImageFile

# frame, row, col = (99, 144, 256)

# batch_size = 15
# num_classes = 2

# row_hidden = 128
# col_hidden = 128

classes=['no_crash','crash']



Model1 = load_model('finalmodel.h5')

pred=['crash_cut/crash/7_10.mp4']

count =0

# vid = cv2.VideoCapture(pred[0])
# ret = True
# while ret:
#     if ret == True:
#         d = make_dataset(pred)
#         ret, frame = vid.read()
#         try:
#             img = Image.fromarray(frame)
#         except ValueError:
#             break
#         except AttributeError:
#             break  
#         predicted = Model1.predict(img)          
#         index = int(predicted.item())
#         if index == 0:
#             cv2.imwrite(r"img/frame%d.png" % count, frame)
#             count += 1
#         else:
#             cv2.imwrite(r"img1/frame%d.png" % count, frame)

#         labels1 = 'status: ' + classes[index]
#         labels2 = 'accuracy: ' + str(100*int(score = tf.nn.softmax(predicted[0])))
            

#         cv2.putText(frame, labels1, (10, 100),
#                     cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
#         cv2.putText(frame, labels2, (10, 200),
#                     cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
#         cv2.imshow('Frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# vid.release()
# cv2.destroyAllWindows()



def pred_model(file_path):    
    cap = cv2.VideoCapture(file_path[0])  
    print(predictions)

    phantram = 0
    true = "no crash"  
    
    while True:
        success, img = cap.read()   

        if success:
                             
            cv2.putText(img, str( "Accuracy:" + str(int(phantram))), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

            cv2.putText(img, str( "State:" + str(true)), (528, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 0), 3)

            true = classes[np.argmax(score)]
            phantram = 100 * np.max(score) 

            cv2.imshow("Image", img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                 break
           
        else:
            break


if __name__ == "__main__":
    d = make_dataset(pred)                         ### can load all of valid/test datasets at once in memory
    predictions = Model1.predict(d)
    score = tf.nn.softmax(predictions[0])
    pred_model(pred)