"""
Metadata returned from model.invoke()
=======================================
model.invoke("...") never returns a plain string. It returns an AIMessage object
(langchain_core.messages.AIMessage), and the text you usually print is just one of
several fields on it. The rest is metadata: token counts, why generation stopped,
a trace id, tool calls the model wants to make, etc.

Below is every field on AIMessage, with real values captured from an actual
ChatGoogleGenerativeAI.invoke() call (see the repr right under this docstring),
followed by which ones you actually need to look at day-to-day.

Real example repr (gemini-2.5-flash, prompt="What is the capital of France?"):

AIMessage(
    content='The capital of France is **Paris**.',
    additional_kwargs={},
    response_metadata={
        'finish_reason': 'STOP',
        'model_name': 'gemini-2.5-flash',
        'safety_ratings': [],
        'model_provider': 'google_genai',
    },
    id='lc_run--01a0fb41-70d4-75f3-8817-87758566fa18-0',
    tool_calls=[],
    invalid_tool_calls=[],
    usage_metadata={
        'input_tokens': 8,
        'output_tokens': 29,
        'total_tokens': 37,
        'input_token_details': {'cache_read': 0},
        'output_token_details': {'reasoning': 21},
    },
)
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=google_api_key)

result = model.invoke("What is the capital of France?")


# =============================================================================
# 1) content  -- the actual answer text
# =============================================================================
# str, or occasionally list[str | dict] for multimodal/multi-part responses.
# This is the only field most tutorials show you.

print(result.content)
# -> 'The capital of France is **Paris**.'


# =============================================================================
# 2) response_metadata  --  **important** -- info about the generation itself
# =============================================================================
# A dict whose exact keys are provider-specific (Gemini's shape differs from
# OpenAI's, Anthropic's, etc.) -- always print it once for a new provider to see
# what you actually get. For ChatGoogleGenerativeAI it currently contains:
#
#   finish_reason   -- **important** -- why generation stopped. Check this before
#                       trusting a response is complete.
#                       'STOP'                    -> finished normally
#                       'MAX_TOKENS'               -> got cut off, hit max_output_tokens
#                       'SAFETY'                    -> blocked by safety filters
#                       'RECITATION'                -> blocked for reciting training data
#                       'OTHER'                      -> unspecified
#   model_name       -- which model actually served the request
#   model_provider    -- 'google_genai' (useful if you swap providers at runtime)
#   safety_ratings    -- list of per-category safety scores for the response

print(result.response_metadata)
# -> {'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'}

if result.response_metadata.get("finish_reason") == "MAX_TOKENS":
    print("Warning: response was truncated, raise max_output_tokens")


# =============================================================================
# 3) usage_metadata  --  **important** -- token counts, i.e. what you're billed for
# =============================================================================
# A dict (UsageMetadata) with this shape:
#
#   input_tokens          -- tokens in your prompt
#   output_tokens         -- tokens the model generated
#   total_tokens          -- input + output
#   input_token_details   -- breakdown dict: {'audio': int, 'cache_creation': int, 'cache_read': int}
#                             (keys only appear if non-zero / applicable to the provider)
#   output_token_details  -- breakdown dict: {'audio': int, 'reasoning': int}
#                             'reasoning' = tokens spent on internal "thinking" (2.5 models)
#                             -- you're often billed for these even though you never see them in `content`.

print(result.usage_metadata)
# -> {'input_tokens': 8, 'output_tokens': 29, 'total_tokens': 37,
#     'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 21}}

print(f"This call used {result.usage_metadata['total_tokens']} tokens "
      f"({result.usage_metadata['output_token_details'].get('reasoning', 0)} of them on reasoning)")


# =============================================================================
# 4) id  --  trace id for this generation
# =============================================================================
# str. Useful for logging/debugging -- correlate a logged error with the exact
# generation call, or match it up in LangSmith traces.

print(result.id)
# -> 'lc_run--01a0fb41-70d4-75f3-8817-87758566fa18-0'


# =============================================================================
# 5) tool_calls / invalid_tool_calls  --  **important for agents/tool use**
# =============================================================================
# Only populated when the model was bound to tools (model.bind_tools([...])) and
# decided to call one instead of just answering in `content`.
#
# tool_calls: list of dicts shaped like:
#   {'name': str, 'args': dict[str, Any], 'id': str | None, 'type': 'tool_call'}
#
# invalid_tool_calls: list of dicts for calls that failed to parse correctly:
#   {'name': str | None, 'args': str | None, 'id': str | None,
#    'error': str | None, 'type': 'invalid_tool_call', 'index': int | str, 'extras': dict}
#
# Both are empty lists ([]) when no tools are involved, like in this example.

print(result.tool_calls)        # -> []
print(result.invalid_tool_calls)  # -> []

for call in result.tool_calls:
    print(call["name"], call["args"])  # how you'd actually dispatch a tool call


# =============================================================================
# 6) additional_kwargs  --  provider-specific extras that don't fit elsewhere
# =============================================================================
# dict, provider-dependent. For OpenAI this is where raw function_call / refusal
# fields have historically shown up. Empty for this Gemini call.

print(result.additional_kwargs)
# -> {}


# =============================================================================
# 7) type / name  --  message-role bookkeeping, rarely useful directly
# =============================================================================
# type: always the literal 'ai' for AIMessage (distinguishes it from 'human'/'system'
#       messages when you're looking at a list of messages, e.g. chat history).
# name: optional human-readable name for the message; None unless you set it yourself.

print(result.type, result.name)
# -> ai None


# =============================================================================
# Quick reference: what to actually check in real code
# =============================================================================
# Always look at                -> result.content
# Before trusting completeness  -> result.response_metadata["finish_reason"]
# For cost/usage tracking       -> result.usage_metadata["total_tokens"] (and the *_token_details breakdowns)
# For debugging/tracing         -> result.id
# If using tools/agents         -> result.tool_calls (and invalid_tool_calls for failures)
# Everything else               -> additional_kwargs, type, name -- rarely needed day-to-day
