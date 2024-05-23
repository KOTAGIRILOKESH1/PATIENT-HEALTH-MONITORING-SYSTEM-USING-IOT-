#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
MAX30105 particleSensor;
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#define ONE_WIRE_BUS D3
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
#include "ThingSpeak.h"
#include <ESP8266WiFi.h>
int statusCode = 0;
WiFiClient  client;
const int FieldNumber1 = 1;
String strs[14]={"2470635","I0FWOVY98BIRMPVW","0","0","SRC 24G","src@internet","0","0","0","0","0","0","0","0"};
int StringCount = 0;

int xval,yval,zval,magnitude;
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

const byte RATE_SIZE = 4; //Increase this for more averaging. 4 is good.
byte rates[RATE_SIZE]; //Array of heart rates
byte rateSpot = 0;
long lastBeat = 0; //Time at which the last beat occurred
float beatsPerMinute;
int beatAvg;
int tempF,spo2;
int led=D4;
int fall=0;
int buz=D5;
int fst=0;
long int prv=0,prv1=0;
int abnormal=0;
void setup()
{
  Serial.begin(9600);
  
  particleSensor.begin(Wire, I2C_SPEED_FAST);
  particleSensor.setup(); //Configure sensor with default settings
  particleSensor.setPulseAmplitudeRed(0x0A); //Turn Red LED to low to indicate sensor is running
  particleSensor.setPulseAmplitudeGreen(0); //Turn off Green LED
  particleSensor.enableDIETEMPRDY();
  sensors.begin();
 accel.begin();
  WiFi.mode(WIFI_STA);
  ThingSpeak.begin(client);
 
  pinMode(led,OUTPUT); 
  delay(1000);
  digitalWrite(led,1);

  pinMode(buz,OUTPUT);
  //Serial.println("WELCOME");
  
}

void loop()
{

  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.print(".");
    delay(1000);
    WiFi.begin(strs[4], strs[5]);
      for(int kk=0;kk<10;kk++)
      {
        digitalWrite(led,0);
        delay(300);
        digitalWrite(led,1);
        delay(300);
      }
   if(WiFi.status() == WL_CONNECTED)
    Serial.println("ok");
  }


  
  long irValue = particleSensor.getIR();
  //Serial.println(irValue);
  if (irValue < 50000)
  {
    if(fst==1)
    {
    fst=0;
    
    beatAvg=0;
    }
  }
  else
  {
   if(fst==0)
   {
    fst=1;
   }

  if (checkForBeat(irValue) == true)
  {
    long delta = millis() - lastBeat;
    lastBeat = millis();
    beatsPerMinute = 60 / (delta / 1000.0);
    if (beatsPerMinute < 255 && beatsPerMinute > 20)
    {
      rates[rateSpot++] = (byte)beatsPerMinute; //Store this reading in the array
      rateSpot %= RATE_SIZE; //Wrap variable
      beatAvg = 0;
      for (byte x = 0 ; x < RATE_SIZE ; x++)
        beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }
  }
  if(beatAvg>40)
  {
  
  tempF = particleSensor.readTemperatureF();
  if(beatAvg<40)
    spo2=0;
  else if(beatAvg>140)
    spo2=map(beatAvg,140,255,80,50);
  else if(beatAvg>=40 && beatAvg<60)
    spo2=map(beatAvg,40,60,70,98);
  else
    spo2=map(beatAvg,60,140,100,80);
  
  }

  else
  {
    tempF = particleSensor.readTemperatureF();
    beatAvg=0;
    spo2=0;
  }

sensors_event_t event; 
 accel.getEvent(&event);
  xval=event.acceleration.x; 
 yval=event.acceleration.y;
  

  fall=0;
  if(xval>5 || xval<-5 || yval>5 || yval<-5 )
  {
    fall=1;
  }

 

  if(millis()-prv>30000)
  {
   
 
    prv=millis();
    sensors.requestTemperatures(); 
    float temperatureF = ((sensors.getTempCByIndex(0)*1.8)+32); 
    Serial.println(String(beatAvg)+","+String(spo2)+","+String(temperatureF)+","+String(fall)+",\n");
  ThingSpeak.setField(1, String(beatAvg));
 ThingSpeak.setField(2, String(spo2));
 ThingSpeak.setField(3, String(temperatureF));
 ThingSpeak.setField(4, String(fall));
 ThingSpeak.setField(5, String(0));
 
 const char* string0 = strs[0].c_str();
 const char* string1 = strs[1].c_str();
 int x = ThingSpeak.writeFields(atol(string0), string1);
 if(x == 200){
  delay(10);
  prv1 = millis();
  }
else{
  delay(10);
  }
  }
}