import os, time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite"

def ask_llm(prompt, system="You are a helpful data analyst.", max_attempts=6):
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system),
            )
            return response.text
        except errors.ServerError:
            wait = 5 * (attempt + 1)
        except errors.ClientError as e:
            if e.code != 429:
                raise
            wait = 10 * (attempt + 1)
        print(f"[llm] busy, retrying in {wait}s (attempt {attempt + 1}/{max_attempts})")
        time.sleep(wait)
    raise RuntimeError("LLM still unavailable after several retries")

if __name__ == "__main__":
    print(ask_llm("Write pandas code to print the number of rows in df. Return only code."))
