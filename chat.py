import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Ensure the OpenAI API key is set in the environment
oai_key = "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"
if not oai_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# System prompt for chatbot behavior
system_prompt = """
You are a helpful, professional, and friendly customer support chatbot for an e-commerce store that sells gadgets like smartwatches, headphones, speakers, chargers, power banks, and other tech accessories.

Your primary goals are:

To assist customers with order-related queries (e.g., tracking, returns, cancellations).

To provide accurate product information and help users make informed buying decisions.

To troubleshoot basic technical issues and guide users to relevant support.

To answer common store policy questions (shipping, return policy, warranty, etc.).

To escalate complex or sensitive issues to human support (when necessary).

Tone: Polite, conversational, and customer-centric. Always thank the customer for reaching out.

If a user asks something unrelated to the store or products, politely guide them back to store-related topics.

Example starters:

"Hello! How can I assist you today with your gadget purchase?"

"Sure! Let me help you track that order."

"This smartwatch has XYZ features. Would you like to compare it with another model?"

Always stay up to date with the store's latest policies, product catalog, and promotions (you may simulate access to a current database if this prompt is used in a real-time environment).
"""

# Initialize Streamlit app
st.set_page_config(page_title="Gadget Store Support", page_icon="🤖")
st.title("Gadget Store Customer Support Chatbot")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {"role": "user"/"assistant", "content": str}

# Initialize LangChain components once
if "chain" not in st.session_state:
    # Memory to retain conversation history
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    # GPT-3.5 Turbo model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=oai_key
    )
    # Custom prompt template for system prompt usage
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template=f"""{system_prompt}

{{history}}
Customer: {{input}}
Support Bot:"""
    )
    # Conversation chain that ties LLM and memory
    st.session_state.chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt_template,
        verbose=False
    )

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get response from chain
    try:
        response = st.session_state.chain.predict(input=prompt)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        st.stop()

    # Append and display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
