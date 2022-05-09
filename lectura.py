import cv2
cap = cv2.VideoCapture("rtsp://admin:Tgestiona2022@169.254.77.132:554/trackID=1")

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()