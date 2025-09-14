import json
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ
with open("faq.json", "r") as f:
    faq = json.load(f)

questions = faq["questions"]
answers = faq["answers"]

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(questions)

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Response function
def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()
    if similarity[0][idx] > 0.3:  # similarity threshold
        return answers[idx]
    else:
        return "Sorry, I don't understand your question. Please contact support."

# Speech recognizer
recognizer = sr.Recognizer()
mic = sr.Microphone()  # Works with sounddevice backend

# Greet user
speak("Hello! I am your voice assistant. You can ask me questions now.")
print("Say 'exit', 'quit', or 'bye' to stop.")

while True:
    try:
        with mic as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source)

        query = recognizer.recognize_google(audio)
        print("You:", query)

        if query.strip().lower() in ["exit", "quit", "bye"]:
            speak("Goodbye! Have a nice day.")
            break

        response = get_response(query)
        print("Bot:", response)
        speak(response)

    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        speak("Sorry, I did not understand that.")
    except sr.RequestError:
        print("Speech service is unavailable.")
        speak("Speech service is unavailable.")
    except KeyboardInterrupt:
        print("\nExiting...")
        speak("Goodbye! Have a nice day.")
        break
