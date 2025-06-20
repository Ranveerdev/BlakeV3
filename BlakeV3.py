# voice_assistant.py
import pyttsx3
import wikipedia
import speech_recognition as sr
import function_word

class VoiceAssistant:
    """Simple Wikipedia voice‑query assistant."""

    def __init__(self, tts_rate: int = 125):
        # --- Speech & TTS engines ----------------------------------------
        self.recognizer = sr.Recognizer()
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", tts_rate)

        # --- Startup message ---------------------------------------------
        print("Blake initiated")
        self.speak("Blake initiated and ready.")

    # ---------------------------------------------------------------------
    # Core I/O helpers
    # ---------------------------------------------------------------------
    def speak(self, text: str) -> None:
        """Say something aloud and print it."""
        print(f"[Blake] {text}")
        self.tts.say(text)
        self.tts.runAndWait()

    def listen(self) -> str | None:
        """Capture a single utterance from microphone and return lowercase text."""
        with sr.Microphone() as source:
            self.speak("I am listening, sir.")
            audio = self.recognizer.listen(source)

        try:
            return self.recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            self.speak("Sorry, I did not catch that.")
        except sr.RequestError as e:
            self.speak("Speech service is unavailable.")
            print("Speech API error:", e)
        return None

    # ---------------------------------------------------------------------
    # Command processing
    # ---------------------------------------------------------------------
    def handle_wikipedia(self, token_list: list[str]) -> None:
        """Process a Wikipedia query and speak back two‑sentence summary."""
        # Strip command words
        for word in ("wikipedia", "search"):
            while word in token_list:
                token_list.remove(word)

        query_tokens = function_word.remove_function_word(token_list)
        if not query_tokens:
            self.speak("I need more keywords for a Wikipedia search.")
            return

        search_query = " ".join(query_tokens)
        print("Wikipedia search query:", search_query)

        try:
            summary = wikipedia.summary(search_query, sentences=2)
            self.speak(summary)
        except Exception as e:
            print("Wikipedia error:", e)
            self.speak("Wikipedia lookup failed.")

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------
    def run(self) -> None:
        """Main REPL loop; Ctrl+C to quit."""
        try:
            while True:
                utterance = self.listen()
                if not utterance:
                    continue

                print("Text:", utterance)
                tokens = utterance.split()
                print("Tokens:", tokens)

                if {"wikipedia", "search"} & set(tokens):
                    self.handle_wikipedia(tokens)
                else:
                    self.speak("No Wikipedia search command detected.")

        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C)")
            self.speak("Goodbye, sir.")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
