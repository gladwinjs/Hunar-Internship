# Simple Rule-Based Chatbot with Enhanced Rules

def chatbot_response(user_input):
    # Convert input to lowercase for case-insensitive matching
    user_input = user_input.lower()
    
    # Rule-based responses
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I assist you today?"
    elif "your name" in user_input:
        return "I am a simple rule-based chatbot created to assist you!"
    elif "help" in user_input:
        return ("Sure! You can ask me about the weather, time, date, a joke, "
                "or just say 'hello'. If you want to end the chat, just type 'bye'.")
    elif "time" in user_input:
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time}."
    elif "date" in user_input:
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        return f"Today's date is {current_date}."
    elif "weather" in user_input:
        return "I can't check the weather right now, but it's always a good idea to carry an umbrella!"
    elif "joke" in user_input:
        return "Why don't scientists trust atoms? Because they make up everything!"
    elif "thank you" in user_input or "thanks" in user_input:
        return "You're welcome! I'm here to help."
    elif "how are you" in user_input:
        return "I'm just a bunch of code, but I'm doing great! How about you?"
    elif "bye" in user_input or "goodbye" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "I'm sorry, I don't understand that. Can you please rephrase or ask something else?"

# Main loop to interact with the chatbot
if __name__ == "__main__":
    print("Welcome to the Enhanced Rule-Based Chatbot! Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "goodbye"]:
            print("Chatbot: Goodbye! Have a nice day.")
            break
        response = chatbot_response(user_input)
        print("Chatbot:", response)
