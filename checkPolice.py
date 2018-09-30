import pyrebase
import cv2
import winsound
i=1

config = {
  "apiKey": "apiKey",
  "authDomain": "bageera2018.firebaseapp.com",
  "databaseURL": "https://bageera2018.firebaseio.com",
  "storageBucket": "bageera2018.appspot.com",
  #"serviceAccount": "path/to/serviceAccountCredentials.json"
}

def stream_handler(message):
    global i
    if i != 1:
        #print(message)
        #users = db.child("users").get()
        #print(users.val())
        users=message["data"]
        check=(list(users.values()))
        dlink=check[0]
        print(dlink)
        storage.child(dlink).download(r"C:\Users\JASPREET SINGH\Desktop\downloaded.jpeg")
        #print(img)
        #mat = cv2.imread('downloaded.jpeg',0)
        #print(mat)
        #cv2.imshow('downloaded.jpeg',mat)
        a=cv2.imread(r'C:\Users\JASPREET SINGH\Desktop\downloaded.jpeg')
        #print(a)
        #sound = pyttsx.init()
        #sound.say('Emergency')
        #sound.runAndWait()
        
        cv2.imshow('check',a)
    else:
        i=2
        
firebase = pyrebase.initialize_app(config)
#fb=pyrebase.initialize_app(config2)
storage = firebase.storage()
#update=fb.database()

#storage.child("example.jpeg").put("thumbDiv.jpeg")
#db.child("users").set({1:"example.jpeg"})

db = firebase.database()
my_stream = db.child("users").stream(stream_handler)

#users = db.child("users").get()
#dlink=users.val()
#print(dlink[1])
#storage.child(dlink[1]).download("downloaded.jpg")

