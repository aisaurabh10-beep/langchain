"""
with_structured_output() cheat sheet
=====================================
Closed-source models (OpenAI, Gemini, DeepSeek, Claude, ...) don't let you hook into
their decoding loop the way local/open-weight models do (no logits-level grammar
constraints). Instead, each provider exposes its own structured-output mechanism
over the API: function/tool calling, a "JSON mode" flag, or a JSON-Schema-constrained
response format.

model.with_structured_output(schema, ...) is LangChain's single interface over all of
that. You give it a schema (TypedDict / Pydantic class / raw JSON schema dict), it
converts that schema into whatever the provider needs, sends it, and parses the
reply back into a Python object for you. The returned object is a Runnable, so you
still call .invoke() / .batch() / .stream() on it like any other chain step.
"""

from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import os

load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=google_api_key)


# =============================================================================
# 1) PYDANTIC schema  -- recommended default
# =============================================================================
# - Gives you real validation (wrong type / missing required field -> raises).
# - .invoke() returns an actual instance of your class (dot access, type-checked).
# - Field(description=...) is NOT just documentation: it gets sent to the model
#   as part of the schema and genuinely steers what it writes into that field.

class Review(BaseModel):
    """Structured breakdown of a product review."""  # becomes the schema's description

    summary: str = Field(description="One sentence summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment")
    pros: list[str] = Field(description="Positive points mentioned")
    cons: list[str] = Field(default_factory=list, description="Negative points mentioned")
    rating_out_of_10: Optional[int] = Field(default=None, description="Rating, only if explicitly stated")


pydantic_model = model.with_structured_output(Review)

result = pydantic_model.invoke(
    "The battery life on this laptop is amazing and the screen is gorgeous, "
    "but it runs hot under load and the fan is noisy. I'd give it a 7/10."
)

print(result)               # Review(summary=..., sentiment='positive', pros=[...], cons=[...], rating_out_of_10=7)
print(type(result))         # <class '__main__.Review'>
print(result.sentiment)     # dot access, autocompletes, type-checked


# =============================================================================
# 2) TYPEDDICT schema  -- lightweight, no validation
# =============================================================================
# - Same idea, but output is a plain dict, not a validated object.
# - Nothing stops the model from returning a wrong type into a field; LangChain
#   does not validate TypedDict output the way it validates Pydantic output.
# - Useful when you don't want the Pydantic dependency / overhead for a quick schema.

class ReviewDict(TypedDict):
    """Structured breakdown of a product review."""

    summary: str
    sentiment: Literal["positive", "negative", "neutral"]
    pros: list[str]
    cons: list[str]
    rating_out_of_10: Optional[int]


# typeddict_model = model.with_structured_output(ReviewDict)
# result = typeddict_model.invoke("...")
# print(result)              # -> plain dict: {'summary': ..., 'sentiment': ..., 'pros': [...], ...}
# print(result["summary"])   # dict-style access only


# =============================================================================
# 3) JSON SCHEMA (raw dict)  -- for schemas built dynamically at runtime
# =============================================================================
# - Use this when the schema isn't a fixed Python class: e.g. built from a form,
#   a config file, or someone else's JSON Schema you were handed.
# - Output is a plain dict, matching the schema. Validation depends entirely on
#   what the provider's API enforces, not on LangChain/Pydantic.

json_schema = {
    "title": "Review",
    "description": "Structured breakdown of a product review",
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One sentence summary of the review"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "pros": {"type": "array", "items": {"type": "string"}},
        "cons": {"type": "array", "items": {"type": "string"}},
        "rating_out_of_10": {"type": "integer"},
    },
    "required": ["summary", "sentiment", "pros"],
}

# json_schema_model = model.with_structured_output(json_schema)
# result = json_schema_model.invoke("...")
# print(result)              # -> plain dict shaped like the schema above


# =============================================================================
# 4) with_structured_output() parameters
# =============================================================================
# Full signature (base class, langchain_core.language_models.chat_models.BaseChatModel):
#
#   model.with_structured_output(
#       schema: dict | type,
#       *,
#       include_raw: bool = False,
#       **kwargs,
#   )
#
# schema
#   The TypedDict / Pydantic class / JSON-schema dict from sections 1-3 above.
#   This is the only required argument.
#
# include_raw
#   False (default) -> .invoke() returns just the parsed object (Pydantic instance
#                       or dict). If parsing/validation fails, the exception is raised.
#   True             -> .invoke() always returns a dict with 3 keys:
#                           {"raw": AIMessage, "parsed": <object or None>, "parsing_error": <Exception or None>}
#                       Use this when you want to inspect the raw model reply, or
#                       handle parse failures yourself instead of letting them raise.
#
# **kwargs
#   Passed straight through to the provider-specific override below. The base
#   class itself raises ValueError on unknown kwargs, NotImplementedError if the
#   model doesn't support structured output at all.

include_raw_model = model.with_structured_output(Review, include_raw=True)
raw_result = include_raw_model.invoke("Decent phone. Good camera, mediocre battery.")
print(raw_result["parsed"])         # Review instance, or None if parsing failed
print(raw_result["parsing_error"])  # None, or the exception that was caught
# print(raw_result["raw"])          # the original AIMessage (with tool_calls / content)


# =============================================================================
# 5) Provider-specific kwargs (passed via **kwargs above)
# =============================================================================
# Each provider overrides with_structured_output and adds its own extra knobs on
# top of (schema, include_raw). These are the ones actually installed in this
# project's .venv (inspect yours with `inspect.signature(YourChatModel.with_structured_output)`).
#
# ChatGoogleGenerativeAI (langchain_google_genai) -- what `model` is above
#   with_structured_output(schema, method="json_schema", *, include_raw=False, **kwargs)
#   method: "json_schema" (default) | "function_calling" | "json_mode"
#     - "json_schema"      -> Gemini's native constrained-JSON response mode.
#     - "function_calling" -> wraps the schema as a tool/function call instead.
#     - "json_mode"        -> asks for raw JSON back; you must describe the shape
#                              in the prompt yourself, schema is used only to parse.
#
# ChatOpenAI (langchain_openai)
#   with_structured_output(schema=None, *, method="json_schema", include_raw=False,
#                           strict=None, tools=None, **kwargs)
#   method: "json_schema" (default) | "function_calling" | "json_mode"
#   strict: bool | None  -> True enforces exact schema adherence (OpenAI "strict mode"),
#                            only supported with method="json_schema" or "function_calling".
#   tools:  list | None  -> additional tools the model may call alongside the
#                            structured-output tool (advanced; rarely needed).
#
# ChatDeepSeek (langchain_deepseek)
#   with_structured_output(schema=None, *, method="function_calling", include_raw=False,
#                           strict=None, **kwargs)
#   Same shape as ChatOpenAI, but method defaults to "function_calling" instead of
#   "json_schema" (DeepSeek's JSON-schema support is less mature).
#
# method_model = model.with_structured_output(Review, method="function_calling")


# =============================================================================
# 6) Nested models, enums, and optional fields all compose normally
# =============================================================================
# Structured output isn't limited to flat fields -- schemas can nest.

class Author(BaseModel):
    name: str = Field(description="Author's full name")
    affiliation: Optional[str] = Field(default=None, description="University or company")


class PaperSummary(BaseModel):
    """Structured explanation of a research paper."""

    title: str = Field(description="Title of the paper")
    authors: list[Author] = Field(description="List of authors and their affiliations")
    problem: str = Field(description="The problem/question the paper addresses")
    method: str = Field(description="The approach or method used")
    key_findings: list[str] = Field(description="Main results, as short bullet points")
    limitations: list[str] = Field(default_factory=list, description="Limitations the paper itself mentions")


# paper_model = model.with_structured_output(PaperSummary)
# result = paper_model.invoke(f"Explain this paper:\n\n{paper_text}")
# print(result.authors[0].name)   # nested objects, still dot-accessible


# =============================================================================
# Quick decision guide
# =============================================================================
# Need validation + type-safe Python objects back          -> Pydantic   (section 1)
# Just want type hints for the LLM, no validation overhead  -> TypedDict  (section 2)
# Schema is built dynamically / isn't a Python class        -> JSON Schema dict (section 3)
# Want to see the raw AIMessage or handle parse errors       -> include_raw=True (section 4)
# Model's default method is unreliable for your schema      -> try a different method= kwarg (section 5)
