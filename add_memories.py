import asyncio
from mem0 import AsyncMemory
import os
from mem0.configs.base import MemoryConfig
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["DATABASE_URL"]= os.getenv("DATABASE_URL")
os.environ["SUPABASE_URL"]= os.getenv("SUPABASE_URL")
os.environ["SUPABASE_KEY"]=os.getenv("SUPABASE_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

custom_prompt = """
Your primary task is to process conversational messages and extract relevant information to store as comprehensive and useful memories. These memories will be used to personalize future interactions and maintain conversational context. Differentiate between 'user' and 'assistant' roles.

**Overall Goal: Create a rich, evolving understanding of the user and the conversation history to enable highly personalized and coherent future interactions.**

**Aim to capture meaningful and substantial information as single, well-formed memory facts. Prioritize storing complete ideas, preferences, personal details, significant statements, and key information shared, rather than breaking them down into short, fragmented pieces.**

**Processing for 'user' messages:**

For messages where the 'role' is 'user', identify and extract:

*   **Personal Information & Preferences:**
    *   **Explicit Facts:** Name, location (if shared and relevant), occupation, age, family details, pets, etc.
        *   *Example: "My dog, Sparky, loves going to the park." -> "User has a dog named Sparky who enjoys the park."*
    *   **Likes & Dislikes:** Stated preferences for activities, food, music, movies, books, topics, etc.
        *   *Example: "I can't stand horror movies." -> "User dislikes horror movies."*
    *   **Opinions & Beliefs:** Stated viewpoints on various subjects, values, or general philosophies.
        *   *Example: "I think it's important to be kind to everyone." -> "User believes it is important to be kind."*
    *   **Goals & Aspirations:** Things the user wants to achieve, learn, or experience.
        *   *Example: "I'm trying to learn how to play the guitar." -> "User goal: Learning to play the guitar."*
    *   **Past Experiences & Anecdotes:** Significant life events, memories, or stories shared by the user that reveal their background or personality.
        *   *Example: "I once backpacked through Europe for three months." -> "User experience: Backpacking through Europe for three months."*
    *   **Current State/Context (if significant):** How the user is feeling, what they are currently doing, or their immediate environment if it impacts the conversation's context significantly and isn't transient.
        *   *Example: "I'm feeling a bit overwhelmed with work this week." -> "User state: Feeling overwhelmed with work this week." (Useful for checking in later)*
    *   **Interests:** Topics the user frequently discusses, shows curiosity about, or explicitly states as an interest.
        *   *Example: User asks many questions about space exploration. -> "User interest: Space exploration."*

*   **What to AVOID storing from user messages:**
    *   Generic questions or brief queries that do not contain meaningful personal information or context (e.g., "What's the weather like?", "How are you?").
    *   Information that is clearly stated as hypothetical, not reflecting a genuine personal detail or preference.
    *   Fleeting comments that don't contribute to a deeper understanding of the user.
    *   Information the user explicitly asks not to be remembered.

**Processing for 'assistant' messages:**

For messages where the 'role' is 'assistant', identify and extract information that is crucial for maintaining conversational continuity and remembering what significant help or information was provided *to this specific user*:

*   **Key Information/Facts Provided:** Specific, important pieces of information, explanations, or definitions shared by the assistant that the user found helpful or might refer back to.
    *   *Example: Assistant explains a complex concept. -> "Assistant explained [concept X] to the user."*
*   **Recommendations Made:** Specific suggestions for books, movies, activities, resources, etc., especially if the user showed interest.
    *   *Example: "You might enjoy the book 'Sapiens'." -> "Assistant recommended the book 'Sapiens' to the user."*
*   **Instructions or Guidance Given:** Key steps or advice provided for a task the user was trying to accomplish.
    *   *Example: Assistant provides steps to reset a password. -> "Assistant provided steps for password reset: [brief summary of steps]." *
*   **Commitments or Future Actions:** If the assistant promised to do something later (e.g., "I can remind you about that tomorrow.").
    *   *Example: "I'll look that up for you and let you know." -> "Assistant committed to looking up [topic] for the user."*
*   **Clarifications & Shared Understandings:** Important points that were clarified, or agreements reached during the conversation.
    *   *Example: "So, to confirm, you're looking for vegetarian recipes?" -> "Shared understanding: User is looking for vegetarian recipes."*

*   **What to AVOID storing from assistant messages:**
    *   Generic conversational filler ("You're welcome!", "I understand.", "Okay.").
    *   Standard self-introductions or disclaimers (e.g., "I am an AI language model.").
    *   Information that is common knowledge and not specifically tailored or significant to the user's current context.
    *   Repetitive acknowledgments.

**Focus on extracting information that is:**

*   **Personally Revealing (for user messages):** Directly tells you something unique or characteristic about the user.
*   **Contextually Significant:** Important for understanding the flow of the current conversation or will be highly relevant for future interactions with *this* user.
*   **Substantial:** Represents a complete idea, fact, or preference, not just isolated words or trivial details.
*   **Actionable/Memorable (for assistant messages):** Represents a key piece of information, advice, or commitment the AI made that is worth remembering for continuity with this user.

***Note***: The primary goal is to build a useful profile of the user and remember significant assistant contributions *to that user* to enhance personalization and conversational coherence. Avoid storing redundant, trivial, or overly generic information.

**Input Message List Example 1:**
[{"role": "user", "content": "I'm really into sci-fi movies, especially ones by Christopher Nolan. I also love to cook Italian food on weekends."}, {"role": "assistant", "content": "Christopher Nolan is a fantastic director! 'Interstellar' is one of my favorites. Italian food is delicious; do you have a signature dish?"}]
Output: {"facts" : ["User interest: Sci-fi movies, particularly by Christopher Nolan.", "User hobby: Enjoys cooking Italian food on weekends.", "Assistant mentioned 'Interstellar' as a favorite Christopher Nolan movie."]}

**Input Message List Example 2:**
[{"role": "user", "content": "I'm trying to plan a surprise birthday party for my friend Sarah, who loves hiking."}, {"role": "assistant", "content": "That's a thoughtful idea! For a hiking enthusiast, a party at a scenic park or a hiking-themed gathering could be great. Some popular local trails are Eagle Peak and Redwood Creek."}]
Output: {"facts" : ["User goal: Planning a surprise birthday party for friend Sarah.", "User's friend Sarah loves hiking.", "Assistant suggested: Party at a scenic park or hiking-themed gathering for Sarah's birthday.", "Assistant recommended local trails: Eagle Peak and Redwood Creek."]}

**Input Message List Example 3:**
[{"role": "user", "content": "Hi there!"}, {"role": "assistant", "content": "Hello! How can I help you today?"}]
Output: {"facts" : []}

**Input Message List Example 4:**
[{"role": "user", "content": "What's the weather like today?"}, {"role": "assistant", "content": "It's sunny with a high of 75 degrees."}]
Output: {"facts" : []} // (Unless the user elaborates, e.g., "Oh great, I can finally go for that run I've been planning.")

**Input Message List Example 5:**
[{"role": "user", "content": "My name is Alex, and I live in London. I'm struggling to stay motivated to exercise."}, {"role": "assistant", "content": "Hi Alex! It's common to struggle with exercise motivation. Perhaps setting small, achievable goals or finding a workout buddy could help? London has many great parks for outdoor activities."}]
Output: {"facts" : ["User name: Alex.", "User location: Lives in London.", "User challenge: Struggling to stay motivated to exercise.", "Assistant suggestion for motivation: Set small, achievable goals or find a workout buddy.", "Assistant mentioned: London has many great parks for outdoor activities."]}

**Input Message List Example 6:**
[{"role": "user", "content": "I feel that learning a new skill every year is important for personal development. This year, I want to learn basic coding, starting with Python."}, {"role": "assistant", "content": "That's a fantastic goal! Python is an excellent choice for beginners. I can recommend resources like 'Codecademy' or 'Automate the Boring Stuff with Python' to get you started."}]
Output: {"facts" : ["User belief: Learning a new skill every year is important for personal development.", "User goal for this year: Learn basic coding, starting with Python.", "Assistant recommended resources for learning Python: 'Codecademy' and 'Automate the Boring Stuff with Python'."]}
"""

prompt = "For building a personalized conversational memory, prioritize extracting: User's core personal details (e.g., name, location if shared), explicit preferences (likes/dislikes), stated opinions, declared goals or aspirations, significant past experiences or anecdotes shared, and topics of clear, expressed interest. From the assistant's contributions, capture key information provided directly to the user, specific recommendations made, actionable guidance or instructions given, and any explicit commitments made to the user. All extracted facts should be substantial and directly useful for personalizing future dialogue and maintaining conversational context."

custom_config = MemoryConfig(
    llm={
        "provider": "gemini",
        "config": {
            "model": "gemini-2.0-flash-lite",
            "temperature": 0.1,
            "max_tokens": 5000,
        }
    },
    embedder={
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
            "embedding_dims": 768
        }
    },
    vector_store={
        "provider": "supabase",
        "config": {
            "connection_string": os.getenv("DATABASE_URL"),
            "collection_name": "memories",
            "embedding_model_dims": 768
        }
    },  
    custom_prompt=custom_prompt,
    version="v1.1"

)
memory = AsyncMemory(config=custom_config)

async def add_memory(info: str,user_id:str):
        messages = [
            {"role": "user", "content":info },
        ]
        await memory.add(messages, user_id=user_id, prompt=prompt)
        return {"status": "success"}



