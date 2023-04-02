import cv2

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Define the mosaic size
mosaic_size = 3
#
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using the face detector
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face, create a mosaic
    for (x, y, w, h) in faces:
        # Resize the face to the mosaic size
        face_roi = cv2.resize(frame[y:y+h, x:x+w], (mosaic_size, mosaic_size))

        # Resize the mosaic back to the original size
        mosaic = cv2.resize(face_roi, (w, h), interpolation=cv2.INTER_AREA)

        # Replace the face in the original image with the mosaic
        frame[y:y+h, x:x+w] = mosaic

    # Display the resulting image
    cv2.imshow('Mosaic', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
