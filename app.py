import streamlit as st
import json

# Load gesture data from JSON
with open('gestures.json', 'r', encoding='utf-8') as f:
    gesture_data = json.load(f)

# Chatbot response logic
def get_response(user_input):
    user_input = user_input.lower()
    for country, data in gesture_data.items():
        if country.lower() in user_input:
            response = f"🌍 **{country}**\n"

            if "greeting" in user_input:
                response += f"🤝 Greeting: {data.get('greeting', 'Not available.')}\n"
            elif "gesture" in user_input:
                response += f"👐 Gesture: {data.get('gesture', 'Not available.')}\n"
            elif "dining" in user_input or "food" in user_input:
                response += f"🍽️ Dining Etiquette: {data.get('dining', 'Not available.')}\n"
            elif "dress" in user_input or "clothing" in user_input:
                response += f"👗 Dress Code: {data.get('dress', 'Not available.')}\n"
            elif "do" in user_input or "don't" in user_input or "dont" in user_input:
                dos = data.get("dos", [])
                donts = data.get("donts", [])
                response += "✅ Do's:\n"
                response += "\n".join(f"- {item}" for item in dos) if dos else "Not available.\n"
                response += "\n\n❌ Don'ts:\n"
                response += "\n".join(f"- {item}" for item in donts) if donts else "Not available."
            else:
                response += f"""
🤝 Greeting: {data.get('greeting', 'Not available.')}
👐 Gesture: {data.get('gesture', 'Not available.')}
🍽️ Dining: {data.get('dining', 'Not available.')}
👗 Dress: {data.get('dress', 'Not available.')}

✅ Do's:
{chr(10).join(f"- {item}" for item in data.get('dos', []))}

❌ Don'ts:
{chr(10).join(f"- {item}" for item in data.get('donts', []))}
                """
            return response
    return "❓ Sorry, I couldn't find information for that. Please ask about one of these countries: India, Brazil, France, Italy, South Korea, China, Japan, Thailand, or Greece."

# Streamlit UI
st.set_page_config(page_title="Culture Buddy 🌏", page_icon="🤖")
st.title("Culture Buddy Chatbot 🌍")
st.write("Ask me about cultural gestures, etiquette, or traditions of a country.")

user_input = st.text_input("Ask a question:")

if user_input:
    response = get_response(user_input)
    st.markdown(response)
