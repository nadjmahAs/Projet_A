/*
  Quiz avec Détection de Gestes par l'IMU
  Ce programme utilise l'unité inertielle (IMU) intégrée pour capturer les données
  d'accélération et de gyroscope. Une fois un nombre suffisant d'échantillons collectés,
  le programme utilise un modèle TensorFlow Lite (Micro) pour classer le mouvement comme un geste connu.
  
  Nadjmah ALI SOIDIKI
*/

#include <Arduino_LSM9DS1.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model_a.h"

const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 119;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "yes",
  "no"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void welcomeMessage() {
  Serial.println("BIENVENUE DANS CE QUIZ !");
  Serial.println("Répondez par les gestes 'Yes (✓)' ou 'No (X)' à chacune des 3 questions qui vous seront posées.");
}

bool askQuestion(const char* question) {
  float aX, aY, aZ, gX, gY, gZ;
  float scoreYes, scoreNo;  // Variables to store the prediction scores
  Serial.println(question);

  // wait for significant motion
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      // read the acceleration data
      IMU.readAcceleration(aX, aY, aZ);

      // sum up the absolutes
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // check if it's above the threshold
      if (aSum >= accelerationThreshold) {
        // reset the sample read count
        samplesRead = 0;
        break;
      }
    }
  }

  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples) {
    // check if new acceleration AND gyroscope data is available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // normalize the IMU data between 0 to 1 and store in the model's
      // input tensor
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples) {
        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return false;
        }

        // Loop through the output tensor values from the model
        for (int i = 0; i < NUM_GESTURES; i++) {

          // Store the prediction scores in variables
          if (strcmp(GESTURES[i], "yes") == 0) {
            scoreYes = tflOutputTensor->data.f[i];
          } else if (strcmp(GESTURES[i], "no") == 0) {
            scoreNo = tflOutputTensor->data.f[i];
          }
        }

        // Maintenant on peut comparer les scores de "yes" et "no"
        if (scoreYes > scoreNo) {
          Serial.println("Vous avez choisi YES (✓).");
          return true;
        } else {
          Serial.println("Vous avez choisi NO (X).");
          return false;
        }     
      }
    }
  }
  return false;  // Au cas où
}

void conclusionMessage(int score) {
  Serial.println("Quiz terminé !");
  Serial.print("Votre score : ");
  Serial.print(score);
  Serial.print(" sur 3");
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  welcomeMessage(); //On fait apparaitre le message de bienvenue

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }


  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  int score =0; //initialisation du score 
  

  // On pose nos trois questions

  if (askQuestion("Est-ce que Paris est la capitale de la France ?")) {
      // La réponse correct est YES
      score++; // Un point en plus si la fonction return true
  }
  if (askQuestion("Est-ce que Berlin est la capitale du Japon ?") == false) {
      // La réponse correct est NO
      score++; // Un point en plus si la fonction return true
  }
  if (askQuestion("Est-ce que l'Afrique est un pays ?") == false) {
      // La réponse correct est NO
      score++; // Un point en plus si la fonction return true
  }

  // On affiche le score final
  conclusionMessage(score);
  Serial.println();
}