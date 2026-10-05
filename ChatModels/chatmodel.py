from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# model = ChatOpenAI(model_name='gpt-4', temperature = 0.7, openai_api_key=openai_api_key)

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',              # which Gemini model to call | e.g. 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite', 'gemini-2.0-flash'
    google_api_key=google_api_key,         # auth key for the Gemini API | any valid API key string from Google AI Studio
    temperature=0.7,                       # higher = more random/creative, lower = more deterministic | range: 0.0 to 2.0
    max_output_tokens=1000,                # caps response length so it can't run unbounded | range: 1 to the model's max output limit (e.g. up to 65536 for 2.5 models)
    top_p=0.9,                             # nucleus sampling: only sample from top 90% probability mass | range: 0.0 to 1.0
    top_k=40,                              # restricts sampling to the top-k most likely tokens | range: 1 to 40 (positive integer, no hard upper limit enforced)
    frequency_penalty=0.0,                 # discourages repeating the same tokens often | range: -2.0 to 2.0
    presence_penalty=0.0,                  # discourages reusing tokens that already appeared at all | range: -2.0 to 2.0
    n=1,                                   # number of candidate completions to generate | range: 1 and up (positive integer)
    stop=None,                             # sequences that stop generation early when produced | None or a list of up to 5 strings
    seed=42,                               # fixes sampling for reproducible outputs | any integer, or None for non-deterministic
    response_mime_type='text/plain',       # expected output format | 'text/plain', 'application/json', or 'text/x.enum'
    thinking_budget=None,                  # token budget for internal reasoning on 2.5 models | None, -1 (dynamic/auto), 0 (disabled), or an int up to ~24576
    include_thoughts=False,                # whether to return the model's reasoning trace | True or False
    reasoning_effort=None,                 # shortcut instead of a manual thinking_budget | None, 'minimal', 'low', 'medium', or 'high'
    safety_settings=None,                  # override default content-safety blocking thresholds | None, or dict of HarmCategory -> BLOCK_NONE/BLOCK_LOW_AND_ABOVE/BLOCK_MEDIUM_AND_ABOVE/BLOCK_ONLY_HIGH
    timeout=None,                          # max seconds to wait for a response before failing | None (no timeout) or a positive float/int number of seconds
    max_retries=6,                         # retry attempts on transient errors before giving up | range: 0 and up (positive integer)
)

# model = ChatDeepSeek(model='deepseek-chat', temperature=0.7, api_key=deepseek_api_key)

result = model.invoke("What is the capital of France?")

print(result)



