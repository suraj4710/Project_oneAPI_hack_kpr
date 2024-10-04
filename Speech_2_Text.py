import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

def recognize_speech_from_mic():
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        
        while True:
            print("Listening... Speak now!")
            audio = None
            
            try:
                # Listen for speech
                audio = recognizer.listen(source, timeout=5)  # Listen until you stop speaking

                # Recognize speech using Google Web Speech API
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")

            except sr.UnknownValueError:
                print("Sorry, I couldn't understand what you said.")
            except sr.RequestError:
                print("There seems to be an issue with the Google Web Speech API.")
            except sr.WaitTimeoutError:
                print("Listening timed out, no speech detected. Continuing to listen...")
            except KeyboardInterrupt:
                print("\nStopped listening.")
                break

if __name__ == "__main__":
    try:
        recognize_speech_from_mic()
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
