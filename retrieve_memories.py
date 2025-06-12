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
Your primary task is to process a user's current question and retrieve the most relevant stored memories to help construct an effective, personalized, and contextually aware answer.

**Input:**
*   User's Current Question (string)
*   A List of Available Stored Memories (list of strings, where each string is a previously saved memory fact)

**Goal:**
Analyze the User's Current Question and identify which of the Available Stored Memories are most relevant for answering it. The retrieved memories should help in:
1.  **Personalizing the response:** Using known user preferences, past statements, or personal details.
2.  **Providing factual recall:** Referencing information previously discussed or explained.
3.  **Maintaining conversational continuity:** Linking to related topics or past interactions.

**Key Categories of Memories to Prioritize for Retrieval (if relevant to the User's Current Question):**

*   **Directly Related User Information & Preferences:**
    *   Retrieve memories about the user's explicit preferences, opinions, goals, past experiences, or personal details (e.g., name, location, interests) that directly relate to the topic of the user's current question.
    *   *Purpose:* To tailor the answer specifically to the user.
    *   *Example:* If user asks "Any good Italian restaurants around?", retrieve "User lives in London." or "User mentioned they love authentic Neapolitan pizza."

*   **Previously Shared Factual Knowledge (by user or assistant):**
    *   Retrieve memories containing definitions, explanations, facts, or key information that were previously discussed and are relevant to the current question. This includes information the assistant provided or facts the user shared.
    *   *Purpose:* To build upon prior knowledge and avoid repetition, or to recall specific details.
    *   *Example:* If user asks "Can you remind me about that Python library you mentioned?", retrieve "Assistant recommended 'Pandas' for data manipulation in Python."

*   **Past Interactions & Solutions:**
    *   Retrieve memories about previous questions the user asked on a similar topic, solutions provided, or guidance given by the assistant that relates to the current query.
    *   *Purpose:* To provide consistent advice or recall previous problem-solving.
    *   *Example:* If user asks "I'm stuck on this coding problem again," retrieve "User was previously working on a Python script for web scraping." or "Assistant previously provided steps for debugging Python scripts: [summary]."

*   **Relevant Conversational Context & Related Topics:**
    *   Retrieve memories that establish broader context or touch upon topics closely related to the user's current question, even if not a direct answer.
    *   *Purpose:* To ensure the answer fits the flow and acknowledges related themes.
    *   *Example:* If user asks "What are your thoughts on AI ethics?", retrieve "User expressed interest in the future of AI." or "Conversation previously touched on responsible AI development."

**Instructions:**

1.  **Analyze the User's Current Question:** Understand its core topic, entities, intent, and any implicit needs.
2.  **Scan Available Stored Memories:** For each memory, assess its relevance to the User's Current Question based on the categories above.
3.  **Prioritize Relevance and Utility:**
    *   Memories that directly answer or significantly inform the question are most important.
    *   Memories that enable personalization are highly valuable.
    *   More recent relevant memories might sometimes be preferred, but direct topical relevance is key.
4.  **Return a Curated List:** Output a JSON object containing a list of the most relevant memory strings. If no memories are deemed sufficiently relevant, return an empty list. The key for the list should be "retrieved_memories".

**Example Scenario 1:**

*   **User's Current Question:** "I'm looking for a new book to read. Any suggestions for a good mystery?"
*   **Available Stored Memories:**
    *   "User name: Alex."
    *   "User dislikes horror novels."
    *   "User mentioned enjoying Agatha Christie books in the past."
    *   "Assistant previously explained the concept of a 'red herring' in mystery plots."
    *   "User lives in New York."
*   **Expected Output:**
    ```json
    {"retrieved_memories": [
        "User dislikes horror novels.",
        "User mentioned enjoying Agatha Christie books in the past.",
        "Assistant previously explained the concept of a 'red herring' in mystery plots."
    ]}
    ```

**Example Scenario 2:**

*   **User's Current Question:** "Can you remind me what we talked about regarding effective study habits?"
*   **Available Stored Memories:**
    *   "User goal: To learn Spanish."
    *   "Assistant suggested the Pomodoro Technique for focused study."
    *   "User finds it hard to concentrate for long periods."
    *   "User likes to listen to classical music."
*   **Expected Output:**
    ```json
    {"retrieved_memories": [
        "Assistant suggested the Pomodoro Technique for focused study.",
        "User finds it hard to concentrate for long periods.",
        "User goal: To learn Spanish."
    ]}
    ```

**Example Scenario 3:**

*   **User's Current Question:** "Hello!"
*   **Available Stored Memories:**
    *   "User name: Sam."
    *   "User is interested in AI."
*   **Expected Output:**
    ```json
    {"retrieved_memories": []}
    ```
    (A simple greeting might not require deep memory retrieval unless the goal is to personalize the greeting itself, e.g., "Hello Sam! Ready to talk more about AI today?")
"""

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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




async def retrieve_memories(request: str,user_id:str):    
        # Retrieve relevant memories (limited to 10 as specified)
        relevant_memories = await memory.search(query=request, user_id=user_id, limit=10)
        memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
        return {"memories": str(memories_str)}



