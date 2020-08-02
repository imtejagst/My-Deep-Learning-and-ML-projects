import cv2
#main image file
img_file = "car_image.jpg"
video = cv2.VideoCapture('tesla dashcam.mp4')

#pretrained car classifier
classifier_file = "car_detector.xml"

#creating car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#run forever
while True:
    read_succsssful, frame = video.read()
    if read_succsssful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break


#detect cars
cars = car_tracker.detectMultiScale(grayscaled_frame)

#Drawing rectangles around the cars
for(x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)





#reading image to opencv
img = cv2.imread(img_file)

#creating car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#convert to grayscale image
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect cars
cars = car_tracker.detectMultiScale(black_n_white)

#print(cars)

#Drawing rectangles around the cars
for(x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)





cv2.imshow('Saiteja car detector', img)

cv2.waitKey()


#print("executed perfectly")