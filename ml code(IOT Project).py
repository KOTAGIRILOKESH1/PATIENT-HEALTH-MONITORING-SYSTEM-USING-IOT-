import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, make_scorer, precision_score
import serial
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import time
import urllib.request
import serial

ascaler = StandardScaler()

ser =serial.Serial('com4',baudrate=9600,timeout=0.1)


def send_sms(phno,msg):
    print("sending SMS..")

    cmd='AT\r\n'
    ser.write(cmd.encode())
    time.sleep(2)
    
    
    cmd='AT+CMGF=1\r\n'
    ser.write(cmd.encode())
    time.sleep(2)
    
                                                
   # phno="9705402407"                          
    cmd='AT+CMGS="'+str(phno)+'"\r\n'
    ser.write(cmd.encode())
   
                           
    time.sleep(1)
    cmd=msg
    ser.write(cmd.encode())  # Message
    
    time.sleep(1)
    cmd = "\x1A"
    ser.write(cmd.encode()) # Enable to send SMS
    time.sleep(10)
    print('SMS Sent')
    time.sleep(1)

           

def takeInput(phno):
    
    while True:


        r_link='https://api.thingspeak.com/channels/2470635/fields/1/last?results=2'
        f=urllib.request.urlopen(r_link)
        pr2 = (f.readline()).decode()

        r_link='https://api.thingspeak.com/channels/2470635/fields/2/last?results=2'
        f=urllib.request.urlopen(r_link)
        pr3 = (f.readline()).decode()

        r_link='https://api.thingspeak.com/channels/2470635/fields/3/last?results=2'
        f=urllib.request.urlopen(r_link)
        pr1 = (f.readline()).decode()

        r_link='https://api.thingspeak.com/channels/2470635/fields/4/last?results=2'
        f=urllib.request.urlopen(r_link)
        pr4 = (f.readline()).decode()
        
        print('TEMP:'+str(pr1)+ ' HB:'+str(pr2) +' SP:'+str(pr3) +' Fall:'+str(pr4) )
        data=str(pr1)+','+str(pr2)+','+str(pr3)+','+str(pr4)
        
        if(data is not None):
            X = np.array([data.split(',')], dtype=np.float32)
            #X = scaler.transform(X)
            print(X)
            # Make a prediction
            y_pred = knn_classifier.predict(X)
           
            print(y_pred)
            
            if y_pred == 1:
                print('Normal')
                send_sms(phno,'Normal')
            if y_pred == 2:
                print('Abnormal')
                send_sms(phno,'Abnormal')
            if y_pred == 3:
                print('Oxygen levels are low')
                send_sms(phno,'Oxygen levels are low')
                
            if y_pred == 4:
                print('High chance of getting Fever')
                send_sms(phno,'High chance of getting Fever')
            if y_pred == 5:
                print('Abnormality in Patient Movement')
                send_sms(phno,'Abnormality in Patient Movement')
            if y_pred == 6:
                print('High chance of getting Fever and Heart Risk')
                send_sms(phno,'High chance of getting Fever and Heart Risk')
            if y_pred == 7:
                print('High chance of getting Fever and Oxygen levels are low')
                send_sms(phno,'High chance of getting Fever and Oxygen levels are low')
            if y_pred == 8:
                print('High chance of getting Fever and Abnormality in Patient Movement')
                send_sms(phno,'High chance of getting Fever and Abnormality in Patient Movement')
            if y_pred == 9:
                print('High chance of Heart Risk and Oxygen levels are low')
                send_sms(phno,'High chance of Heart Risk and Oxygen levels are low')
            if y_pred == 10:
                print('High chance of Heart Risk and Abnormality in patient Movement')
                send_sms(phno,'High chance of Heart Risk and Abnormality in patient Movement')
            if y_pred == 11:
                print('Oxygen levels are low and Abnormality in Patient Movement')
                send_sms(phno,'Oxygen levels are low and Abnormality in Patient Movement')
            if y_pred == 12:
                print('High chance of Heart Risk, Fever and Oxygen levels are low')
                send_sms(phno,'High chance of Heart Risk, Fever and Oxygen levels are low')
            if y_pred == 13:
                print('High chance of getting Fever, Oxygen levels are low and Abnormality in Patient Movement')
                send_sms(phno,'High chance of getting Fever, Oxygen levels are low and Abnormality in Patient Movement')
            if y_pred == 14:
                print('High chance of getting Fever, Heart Risk and Abnormality in Patient Movement')
                send_sms(phno,'High chance of getting Fever, Heart Risk and Abnormality in Patient Movement')
            if y_pred == 15:
                print('High chance of Heart Risk, Oxygen levels are low and Abnormality in Patient Movement')
                send_sms(phno,'High chance of Heart Risk, Oxygen levels are low and Abnormality in Patient Movement')
                            
            time.sleep(15)
                
phno=input('Enter phno number:')         
data = pd.read_csv("data_csv.csv")

y = data['Severity Level']
X = data.drop(['Health Condition','Severity Level'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
knn_classifier = KNeighborsClassifier(n_neighbors = 4)
knn_classifier.fit(X_train, y_train)

knn_preds = knn_classifier.predict(X_test)
knn_acc = accuracy_score(y_test, knn_preds)



rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)  # You can adjust parameters as needed
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
rf_preds = rf_classifier.predict(X_test)

# Accuracy calculation
rf_acc = accuracy_score(y_test, rf_preds)




# Convert accuracy scores into percentages
knn_acc_percentage = knn_acc * 100
rf_acc_percentage = rf_acc * 100

# Print accuracy percentages
print("Accuracy with KNN: {:.2f}%".format(knn_acc_percentage))
print("Accuracy with Random Forest: {:.2f}%".format(rf_acc_percentage))





# Generate some random binary classification data
y_true = np.random.randint(0, 2, size=100)
y_pred = np.random.randint(0, 2, size=100)

# Calculate accuracy and precision for different threshold values
thresholds = np.linspace(0, 1, num=101)
accuracy = []
precision = []
for t in thresholds:
    y_pred_t = (y_pred >= t).astype(int)
    accuracy.append(accuracy_score(y_true, y_pred_t))
    precision.append(precision_score(y_true, y_pred_t))

# Generate some random binary classification data
y_true = np.random.randint(0, 2, size=100)
y_pred = np.random.randint(0, 2, size=100)

takeInput(phno)
