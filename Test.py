import tkinter as tk 
from tkinter import Message, Text
from tkinter.filedialog import askopenfile
from tkinter.ttk import*
from tkinter import*
import cv2
import os
import numpy as np
import sqlite3  
import datetime 
import time
import shutil
import threading
import easyocr
from PIL import Image, ImageTk
import tkinter.ttk as ttk 
import tkinter.font as font 
from pathlib import Path
import face_recognition
import re
import matplotlib
from tkinter import messagebox
import csv
import speech_recognition as sr



def show_popup1():
    popup = tk.Toplevel(window)
    popup.title("MESSAGE")
    x = window.winfo_rootx() + window.winfo_width() // 2 - 100
    y = window.winfo_rooty() + window.winfo_height() // 2 - 50
    popup.geometry(f"300x150+{x}+{y}")
    tk.Label(popup, text="All files uploaded successfully. Thanks!", padx=20, pady=20).pack()
    popup.after(2000, popup.destroy)
    #tk.Button(popup, text="Close", command=popup.destroy).pack()

def show_popup2():
    popup = tk.Toplevel(window)
    popup.title("MESSAGE")
    x = window.winfo_rootx() + window.winfo_width() // 2 - 100
    y = window.winfo_rooty() + window.winfo_height() // 2 - 50
    popup.geometry(f"300x150+{x}+{y}")
    tk.Label(popup, text="Please click on capture image. Thanks!", padx=20, pady=20).pack()
    popup.after(2500, popup.destroy)

def show_popup(title):
    popup = tk.Toplevel(window)
    popup.title("MESSAGE")
    x = window.winfo_rootx() + window.winfo_width() // 2 - 100
    y = window.winfo_rooty() + window.winfo_height() // 2 - 50
    popup.geometry(f"300x150+{x}+{y}")
    tk.Label(popup, text=title, padx=20, pady=20).pack()
    tk.Button(popup, text="Close", command=popup.destroy).pack()


    
#function for browsing files
def upload_docs():
    #add line--21--09
    file_path = askopenfile(mode='r',title="Select Files", filetypes=[('Image Files','.png;.jpg;.jpeg')])
    if file_path:
        file_name=file_path.name
        print(file_name)
        show_popup('File Uploaded Successfully!'f" Selected file path:\n{file_name}")
        file_names.append(file_name)

#function for uploading files
def submit_docs():
    if not file_names:
        return

    #Getting full name from UI
    full_name=name_txt.get()
    
    destination_folder_main = full_name
    os.makedirs(destination_folder_main)
        
    destination_sub_folder = "Uploaded_documents"
    destination_path = os.path.join(destination_folder_main, destination_sub_folder)
    os.makedirs(destination_path)
    i=0
    ## Values to store in CSV
    aadhaarName="None"
    aadhaarNo="None"
    aadhaarDob="None"
    isAadhaarValid="False"
    
    panNo="None"

    passportNo="None"
    
    
    ########
    
    for file_name in file_names:
        i+=1
        print(i)
        if i == 1:
            file_name_destination=os.path.basename("aadhaar.jpg")
        elif i == 2:    
            file_name_destination=os.path.basename("pan.jpg")
        elif i == 3:    
            file_name_destination=os.path.basename("passport.jpg")
            
                        
        print(file_name_destination)
        destination_path_final = os.path.join(destination_path, file_name_destination)
        shutil.copy(file_name, destination_path_final)
        print(f"Source File '{file_name}' copied successfully")
        print(f"File copied '{destination_path_final}' successfully")
        show_popup1()
        ##Call 1 - Extract text
        extracted_result=process_document_text(destination_path_final,i,destination_folder_main)
        if i == 1:
            #add logic to extract aadhaar card fields and return 
            print("Extracting Aadhaar data")
            # Define a simple regular expression to capture names
            name_pattern = r'\b(?:[A-Z][a-z]+[-\s]?)+[A-Z][a-z]*\b'
            aadhar_pattern = r'^\b\d{4}\s\d{4}\s\d{4}\b$'
            dob_pattern = r'\b(\d{1,2}[-/./]\d{1,2}[-/./]\d{4})\b'
            
            for detection in extracted_result:
                text=detection[1]
                # Search for the pattern in the text
                if re.search(name_pattern, text):
                    aadhaarName=text
                if re.search(aadhar_pattern, text):
                    aadhaarNo=text
                if re.search(dob_pattern, text):
                    aadhaarDob=text
            print(f"Extracted name:",aadhaarName)
            print(f"Extracted aadhar_num:",aadhaarNo)
            print(f"Extracted dob:",aadhaarDob)
            
            
        elif i == 2:
            #add logic to extract pan card fields and return
            print("Extracting PAN data")
            pan_pattern = r'^\b\d{4}\s\d{4}\s\d{4}\b$' #XXXXXnnnnX
            
        elif i == 3:
            #add logic to extract passport card fields and return
            print("Extracting Passport data")

        ##Call 2 - Extract images from documents
        process_document_image(destination_path_final,i,destination_folder_main)

        ##Call 3 - Verify live image against extracted image



        
    ### End of for loop
    ## write all fields to CSV
    
    
    isAadhaarValid = checkAadhaarValidity(aadhaarNo, aadhaarName)
    RecogImg.config(state=tk.NORMAL)      #button will be enabled for recognition
    #RecogImg.config(state=tk.NORMAL)  # button will be enabled for recognition

    retval_submit_docs=aadhaarName+","+aadhaarNo+","+aadhaarDob+","+str(isAadhaarValid)+","+panNo+","+passportNo
    write_to_file(retval_submit_docs)
    #return retval_submit_docs
    #calling process document function to start processing
    #process_document(destination_path_final)
###########################
# Dummy data for the "Aadhaar card" table
aadhaar_database = [
    {"id": "417509511390", "name": "Himanshi Paliwal"},
    {"id": "592782096991", "name": "Shaheen Khan"},
    {"id": "286483829704", "name": "Jenny Stanly"},
    {"id": "485867173934", "name": "Arpit Gupta"},
    {"id": "372545753234", "name": "Hritik Sharma"},
    {"id": "613122616032", "name": "Jhanvi Singh"},
    {"id": "490512293455", "name": "Charu Joshi"},
    {"id": "268483470854", "name": "Nitin Kumar"},
    {"id": "663767376637", "name": "Nikhilesh Ramdas Tonpe"}
    
]
# Himanshi: Function to check adhar validity
def checkAadhaarValidity(matching_aadhar, matching_name):
    for person in aadhaar_database:
        if (person['id'].replace(" ", "") == matching_aadhar.replace(" ", "") and person['name'] == matching_name):
            show_popup("AADHAR VALIDATED SUCCESSFULLY!")
            print(f"{matching_aadhar} is a valid id for {matching_name}.")
            return "True"
    show_popup("INVALID AADHAR CARD!")
    print(f"{matching_aadhar} is an invalid id- {matching_name}.")
    return False



###########################
#NIK: extracting photo from uploaded images
def process_document_image(uploaded_file_path,doc_position,destination_folder_main):
    print("process_document_image -> file_path : "+uploaded_file_path)
    print("process_document_image -> destination_folder_main : "+destination_folder_main)
    
    if doc_position == 1:
        print("aadhaar.jpg")
        destination_subfolder=destination_folder_main+"/extracted_images/aadhaar"
        extracted_img="aadhaar.jpg"
    elif doc_position == 2:    
        print("pan.jpg")
        destination_subfolder=destination_folder_main+"/extracted_images/pan"
        extracted_img="pan.jpg"
    elif doc_position == 3:
        print("passport.jpg")
        destination_subfolder=destination_folder_main+"/extracted_images/passport"
        extracted_img="passport.jpg"
    try:
        #extract image using OpenCV
        doc_img = cv2.imread(uploaded_file_path)
        faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray=cv2.cvtColor(doc_img,cv2.COLOR_BGR2GRAY)
        faces =faceCascade.detectMultiScale(gray, 1.3, 5)
        full_name=name_txt.get()
        #output_folder = destination_folder_main+"/extracted_images"
        output_folder=destination_subfolder
        #Create the output folder 
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        extracted_faces=[]
        for (x,y,w,h) in faces:
            face=doc_img[y:y+h, x:x+w]
            extracted_faces.append(face)
        for i, face in enumerate(extracted_faces):
            image_path = os.path.join(output_folder, extracted_img)
            cv2.imwrite(image_path, face)
            print(f"Image {i} saved: {image_path}")
            ####TrackImages()

    except Exception as e:
        print("Error extracting image")
    
    show_popup2()
        
#############################################################
        
#NIK: extracting text from uploaded images
def process_document_text(uploaded_file_path,doc_position,destination_folder_main):
    print("process_document_text -> file_path : "+uploaded_file_path)
    
        
    reader=easyocr.Reader(['en']) #Language code 'en' for English
    try:
        #extract text using EasyOCR
        result = reader.readtext(uploaded_file_path)
        return result
        
        extracted_text = '\n'.join([text[1] for text in result])
        print("Extracted Text:")
        print(extracted_text)
        
    except Exception as e:
        print("Error extracting text")
        
##############################################################################
#Real-time Facial recognition
def load_known_faces(known_faces_folder):
    print("load_known_faces: known_faces_folder -> "+ known_faces_folder)
    known_faces = []
    for filename in os.listdir(known_faces_folder):
        print("load_known_faces: filename -> "+ filename)
        image = face_recognition.load_image_file(os.path.join(known_faces_folder, filename))
        print("load_known_faces: image.size -> "+ str(image.size))
        print("load_known_faces: image.shape -> "+ str(image.shape))
        face_encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(face_encoding)

    return known_faces


def recognize_faces(known_faces, frame):
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return frame

    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        results = face_recognition.compare_faces(known_faces, face_encoding)
        #display_name = "Unknown"

        if any(results):
            index = results.index(True)
            display_name=name_txt.get()

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, display_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    return frame 

def TrackImages():
    full_name=name_txt.get()
    known_faces_folder = full_name+"/extracted_images/aadhaar"  # Folder containing known faces
    print("TrackImages: known_faces_folder -> "+ known_faces_folder)
    known_faces = load_known_faces(known_faces_folder)
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam, or specify the camera index if multiple cameras are connected.
    
    faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    folder_path="saved_images"
    os.makedirs(folder_path, exist_ok=True)
    #num_images=1
    # Capture and save 'num_images' images

    #for i in range(num_images):
    #    ret, frame = video_capture.read()
    #    image_path = os.path.join(folder_path, f"image_{i}.jpg")
    #    cv2.imwrite(image_path, frame)
    countdown_time=5
    show_countdown_window(countdown_time)
    ret, frame = video_capture.read()
    cv2.imshow('IMAGE', frame)
    cv2.waitKey(0)
    realtime_image = os.path.join(folder_path, f"realtime.jpg")
    cv2.imwrite(realtime_image, frame)
    
    # Release the camera
    video_capture.release()
    cv2.destroyAllWindows()

    # Compare the images
    
    #processed_frame = recognize_faces(known_faces_folder, folder_path)
    compare_images(realtime_image, known_faces_folder)
    
def compare_images(realtime_image, known_faces_folder):
    # Load the reference image containing the face you want to compare
    print("compare_images(): realtime_image : "+realtime_image)
    print("compare_images(): known_faces_folder : "+known_faces_folder)
    reference_image = face_recognition.load_image_file(realtime_image)
    reference_face_encoding = face_recognition.face_encodings(reference_image)[0]

    # List of paths to the multiple images you want to compare
    image_paths = [known_faces_folder+"/aadhaar.jpg"]

    for image_path in image_paths:
        # Load each image
        unknown_image = face_recognition.load_image_file(image_path)

        # Find all faces in the unknown image and encode them
        face_encodings = face_recognition.face_encodings(unknown_image)

        if len(face_encodings) > 0:
            for unknown_face_encoding in face_encodings:
                # Compare each face in the unknown image with the reference face
                results = face_recognition.compare_faces([reference_face_encoding], unknown_face_encoding)

                if results[0]:
                    show_popup("User validated, matched with documents!")
                    print(f"Match found in {image_path}")
                    # Data to append to the same row
                    additional_data = ["Matched"]

                    # Read the existing row from the CSV file
                    with open("UserInfo.csv", mode="r") as file:
                        reader = csv.reader(file)
                        rows = list(reader)

                    # Modify the existing row with the additional data
                    row_to_modify = 1  # Replace 0 with the row number you want to modify
                    modified_row = rows[row_to_modify] + additional_data

                    # Write the modified row back to the CSV file
                    with open("UserInfo.csv", mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(rows[:row_to_modify] + [modified_row] + rows[row_to_modify + 1:])
                    break  # If a match is found, stop processing the current image
                else:
                    show_popup("Your Picture in the document doesn't match with you!")
                    # Data to append to the same row
                    additional_data = ["NOT Matched"]

                    # Read the existing row from the CSV file
                    with open("UserInfo.csv", mode="r") as file:
                        reader = csv.reader(file)
                        rows = list(reader)

                    # Modify the existing row with the additional data
                    row_to_modify = 1  # Replace 0 with the row number you want to modify
                    modified_row = rows[row_to_modify] + additional_data

                    # Write the modified row back to the CSV file
                    with open("UserInfo.csv", mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(rows[:row_to_modify] + [modified_row] + rows[row_to_modify + 1:])
                    print(f"No match found in {image_path}")
        else:
            show_popup(f"No face found in {image_path}")
            print(f"No face found in {image_path}")

def write_to_file(retval_submit_docs_from_submit):
    #write to file
    # Open function to open the file "MyFile1.txt" 
    # (same directory) in read mode and
    file1 = open("UserInfo.csv", "w")
    file1.write("aadhaarName,aadhaarNo,aadhaarDob,isAadhaarValid,panNo,passportNo,ImageValidated?\n")
    file1.write(retval_submit_docs_from_submit)
    file1.close()

        
    retval_submit_docs_from_submit=""

def show_countdown_window(countdown_time):
    window = cv2.namedWindow("Countdown", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Countdown", 300, 150)

    for i in range(countdown_time, 0, -1):
        img = cv2.putText(
            np.zeros((150, 300, 3), dtype=np.uint8),
            str(i),
            (125, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("Countdown", img)
        cv2.waitKey(1000)

    cv2.destroyWindow("Countdown")
### compare_images


# Function to start speech recognition
def start_speech_recognition():

    start_speech_recognition()
        # Initialize the recognizer
    recognizer = sr.Recognizer()


    # Create a VideoCapture object to access the camera (0 is usually the built-in camera)
    #cap = cv2.VideoCapture(0)
 
    #while True:
        # Read a frame from the camera
        # ret, frame = cap.read()

         #if not ret:
            # break
    # Display the frame in a window
    #cv2.imshow('Camera Feed', frame)
        
    # Speech Recognition implementation 
    with sr.Microphone() as source:
        try:    
            audio = recognizer.listen(source)
            recognized_text = recognizer.recognize_google(audio)
            print("You said: " + recognized_text)
        except sr.UnknownValueError:
            recognized_text ="Could not understand audio"
            print(recognized_text)
        except sr.RequestError as e:
            recognized_text ="Error"
            print(recognized_text)

  

    # Compare the strings (case-insensitive)
    if recognized_text.lower() == text_recog.lower():
        #show_popup("SPEECH RECOGNITION SUCCESSFUL !")
        print("Strings are the same")
        #cap.release()
        #cv2.destroyAllWindows()
    else:
        #show_popup("SPEECH RECOGNITION UNSUCCESSFUL !")
        print("Strings are different")
        #cap.release()
        #cv2.destroyAllWindows()


    
        

#Name of the UI window
window =tk.Tk()
window.title("KYC Verification Process")
window.geometry("800x600")

# Center the main window on the screen
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x_coordinate = (screen_width - 800) // 2
y_coordinate = (screen_height - 600) // 2
window.geometry(f"800x600+{x_coordinate}+{y_coordinate}")

# Add a background image or backdrop (replace 'background_image.png' with your image)
background_image = tk.PhotoImage(file="bg3.png")
background_label = tk.Label(window, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Create a frame for the content with a white background
content_frame = tk.Frame(window, bg="#C6E2FF")
content_frame.place(relx=0.13, rely=0.25, relwidth=0.35, relheight=0.53)

# Header Label
heading_frame = tk.Frame(content_frame, bg="#C6E2FF",width=100, height=70)
heading_frame.pack(pady=15, padx=10, fill=tk.X)
logo_image = tk.PhotoImage(file="HSBC4.png")
logo_label = tk.Label(heading_frame, image=logo_image, bg="#C6E2FF")
logo_label.place(relwidth=1, relheight=1)
#header_label = tk.Label(heading_frame, text="Welcome to KYC Process", font=("Helvetica", 20), bg="#C6E2FF")
#header_label.pack(pady=20)

# Create frames for each row of components (Name, Aadhar Card, PAN Card, Passport)
#name_frame = tk.Frame(content_frame, bg="#C6E2FF")
#name_frame.pack(pady=10, padx=50, fill=tk.X)

#aadhar_frame = tk.Frame(content_frame, bg="#C6E2FF")
#aadhar_frame.pack(pady=10, padx=50, fill=tk.X)

#pan_frame = tk.Frame(content_frame, bg="#C6E2FF")
#pan_frame.pack(pady=10, padx=50, fill=tk.X)

#passport_frame = tk.Frame(content_frame, bg="#C6E2FF")
#passport_frame.pack(pady=10, padx=50, fill=tk.X)

# Labels and Entry/Upload Buttons in each row
name_label = tk.Label(window, text="Name", font=("Helvetica", 14),bg="#C6E2FF")
name_label.pack(side=tk.LEFT, padx=10)
name_label.place(x = 250, y = 305)
name_txt = tk.Entry(window, font=("Helvetica", 14))
name_txt.pack(side=tk.RIGHT, padx=10, fill=tk.X, expand=True)
name_txt.place(x = 400, y = 310)

aadhar_label = tk.Label(window, text="Aadhar Card", font=("Helvetica", 14), bg="#C6E2FF")
aadhar_label.pack(side=tk.LEFT, padx=10)
aadhar_label.place(x = 250, y = 365)
#aadhar_file_label = tk.Label(aadhar_frame, text="", font=("Helvetica", 12), bg="#C6E2FF")
#aadhar_file_label.pack(side=tk.LEFT, padx=10)
aadhar_upload_button = Button(window, text="Upload Aadhar Card", width = 18, height = 2, command = lambda:upload_docs())
aadhar_upload_button.pack(side=tk.RIGHT, padx=20)
aadhar_upload_button.place(x = 400, y = 365)

pan_label = tk.Label(window, text="PAN Card", font=("Helvetica", 14), bg="#C6E2FF")
pan_label.pack(side=tk.LEFT, padx=10)
pan_label.place(x = 250, y = 425)
#pan_file_label = tk.Label(window, text="", font=("Helvetica", 12), bg="#C6E2FF")
#pan_file_label.pack(side=tk.LEFT, padx=10)
pan_upload_button = tk.Button(window, text=" Upload PAN Card", width = 16, height = 2, command = lambda:upload_docs())
pan_upload_button.pack(side=tk.RIGHT, padx=20)
pan_upload_button.place(x = 400, y = 425)

#passport_label = tk.Label(window, text="Passport", font=("Helvetica", 14), bg="#C6E2FF")
#passport_label.pack(side=tk.LEFT, padx=10)
#passport_file_label = tk.Label(window, text="", font=("Helvetica", 12), bg="#C6E2FF")
#passport_file_label.pack(side=tk.LEFT, padx=10)
#passport_upload_button = tk.Button(window, text=" Upload Passport ",command = lambda:upload_docs())
#passport_upload_button.pack(side=tk.RIGHT, padx=10)

#button_frame = tk.Frame(content_frame, bg="#C6E2FF")
#button_frame.pack(pady=20, padx=50, fill=tk.X)

# Close Button to exit the application
close_button = tk.Button(window, text="Close the KYC Application", font=("Helvetica", 14), command=window.destroy)
close_button.pack(side=tk.RIGHT,padx=13)
close_button.place(x = 335, y = 560)


# Submit Button
#show_popup
submit_button = tk.Button(window, text="Submit", font=("Helvetica", 14), width = 9, height = 1, command=submit_docs)
submit_button.pack(side=tk.RIGHT,padx=15)
submit_button.place(x = 420, y = 505)


detection_button = tk.Button(window, text="Verify Speech", font=("Helvetica", 14), command= start_speech_recognition)
detection_button.pack(side=tk.RIGHT,padx=15)
detection_button.place(x =550, y = 505)


# Create a label to display the text to be used for verification 
text_recog = "I am a responsible human being"
text_label = tk.Label(window, text= text_recog, font=("Helvetica", 16))
text_label.pack(padx=20, pady=10)

#Facial Recognition
RecogImg = tk.Button(window, text ="Capture Image", command = TrackImages,font=("Helvetica", 14), state=tk.DISABLED)
#font =('times', 15, ' bold '))
RecogImg.place(x = 250, y = 505)

#List to store the uploaded file paths
file_names = []

popup_queue = []

window.mainloop()
