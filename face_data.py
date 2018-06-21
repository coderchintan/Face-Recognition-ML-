
import numpy as np
import cv2

# Initialize camera
cap = cv2.VideoCapture(0)

# Load the haar cascade for frontal face(create the harcascade object to detect images)
face_cascade = cv2.CascadeClassifier('/home/chintan/Downloads/haarcascade_frontalface_alt.xml')

skip = 0 #flag that describe total no of images kitne capture kr liye hai (frame no phli image ka initially zero hai)(current frame number)
face_data = [] #placeholder for storing the data
dataset_path = '/home/chintan/D_Practice/face_dataset/'

file_name = raw_input("Enter the name of the person: ")

while True:
	ret, frame = cap.read()
	if ret == False:
		continue
	# Convert current frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect multi faces in the image (apply the harcascade to detct faces in current frame)
	#the other parameter is used for fine tuning the images of harcascade object
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) #it will detect the image and return an object which consists of multiple points (x,y,w,h)
	k = 1

	faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)

	
    # for each face we extract we have
    #the corner cordinate(y,x)
    #thw w,h for each image
   	for face in faces[:1]:
		x, y, w, h = face

		# Get the face ROI
		offset = 7
		#Get the face section(component) from image frame
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		#resixe the image to (50,50,3) so that every rectangular box has same length and width
		face_section = cv2.resize(face_section, (100, 100))
		
		#
		if skip % 10 == 0:  #har 10-10 frame ke bad  =image ko capture krunga and i have to capture the images less than 20 frames
			face_data.append(face_section)
			print len(face_data)

		# Display the face ROI
		cv2.imshow(str(k), face_section)
		k += 1

		# Draw rectangle in the original image(for visualization) around the faces
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

	# update the frame number
	skip += 1 #increment the current frame

	cv2.imshow("Faces", frame) #Display the frame into new window

	if cv2.waitKey(1) & 0xFF == ord('q'): #camera bndh krne k liye (1) is in ms it will wait for some input,, Bytes format me input read hoga
		break                              # second paranater key q for quit(string ka Ascii value return kr dga)


# Convert face list to numpy array
face_data = np.asarray(face_data)

print face_data.shape #here face_data shape is(n,100,100,3) i.e((5, 100, 100, 3))
face_data = face_data.reshape((face_data.shape[0], -1)) #to make flatten an array heree shape is (n,30000)(-1 represent all the remiaining columns)
print face_data.shape #exapmle((5, 30000))

# Save the dataset in filesystem
np.save(dataset_path + file_name, face_data)
print "Dataset saved at: {}".format(dataset_path + file_name + '.npy') #(Dataset saved at: /home/chintan/D_Practice/face_dataset/random.npy)



cv2.destroyAllWindows() #Destroy all the windows and free or clean the RAM
  
