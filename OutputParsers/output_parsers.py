"""
Output parsers in LangChain
=============================
An output parser takes the raw text a model generates and turns it into a
Python value (a string, list, dict, or Pydantic object). The chain shape is
always the same (LCEL pipe syntax):

    chain = prompt | model | parser
    result = chain.invoke({...})

How this differs from with_structured_output() (see ../StructuredOutput/structured_output.py):
  - with_structured_output() relies on the PROVIDER's native mechanism (function
    calling / JSON mode) to force the shape. It only works with models that
    support one of those.
  - An output parser instead injects plain-text "format instructions" into your
    PROMPT (via parser.get_format_instructions()), asking the model to format its
    reply a certain way, then parses the raw text reply back on the way out.
    This works with ANY chat/completion model, including ones with no native
    structured-output support -- the tradeoff is it's enforced by instruction-
    following, not by the API, so a model can still ignore the formatting and
    cause a parse error.

Use with_structured_output() when your model supports it (cleaner, more robust).
Use an output parser when it doesn't, or when you need a parser in the middle of
a bigger LCEL chain (e.g. prompt | model | parser | another_step).
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from pydantic import BaseModel, Field

load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=google_api_key)


# =============================================================================
# 1) StrOutputParser  -- the one you'll use most often
# =============================================================================
# Does nothing fancy: just pulls `.content` (the string) out of the AIMessage
# that model.invoke() returns, so your chain's output is a plain str instead of
# an AIMessage object. No format instructions needed.

str_chain = model | StrOutputParser()
print(str_chain.invoke("What is the capital of France?"))
# -> 'The capital of France is **Paris**.'


# =============================================================================
# 2) CommaSeparatedListOutputParser  -- quick list output
# =============================================================================
# get_format_instructions() returns text telling the model to reply as
# "item1, item2, item3". parser.parse(...) then splits that string into a
# Python list[str].

csv_parser = CommaSeparatedListOutputParser()

csv_prompt = PromptTemplate(
    template="List 3 {thing}.\n{format_instructions}",
    input_variables=["thing"],
    partial_variables={"format_instructions": csv_parser.get_format_instructions()},
)

csv_chain = csv_prompt | model | csv_parser
print(csv_chain.invoke({"thing": "colors"}))
# -> ['red', 'green', 'blue']


# =============================================================================
# 3) JsonOutputParser  -- arbitrary JSON, no fixed schema required
# =============================================================================
# Without a pydantic_object, it just tells the model "respond with JSON" and
# parses whatever JSON comes back into a dict -- no validation, shape can vary
# call to call. Pass pydantic_object=SomeModel to also steer *which* fields it
# should include (still returns a plain dict, not a validated instance -- for
# that, use PydanticOutputParser below).

json_parser = JsonOutputParser()

json_prompt = PromptTemplate(
    template="Give me the name and population of France as JSON.\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

json_chain = json_prompt | model | json_parser
print(json_chain.invoke({}))
# -> {'name': 'France', 'population': 68373433}


# =============================================================================
# 4) PydanticOutputParser  -- validated, type-safe objects back
# =============================================================================
# get_format_instructions() embeds the actual JSON schema (derived from your
# Pydantic class, including each Field(description=...)) into the prompt.
# parser.parse(...) then both parses AND validates the JSON against your class,
# returning a real instance of it (raises if the model's output doesn't match).

class Country(BaseModel):
    name: str = Field(description="country name")
    capital: str = Field(description="capital city")
    population: int = Field(description="approximate population")


pydantic_parser = PydanticOutputParser(pydantic_object=Country)

pydantic_prompt = PromptTemplate(
    template="Tell me about France.\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
)

pydantic_chain = pydantic_prompt | model | pydantic_parser
result = pydantic_chain.invoke({})
print(result, type(result))
# -> name='France' capital='Paris' population=68000000 <class '__main__.Country'>


# =============================================================================
# 5) XMLOutputParser  -- XML instead of JSON
# =============================================================================
# Same idea as JsonOutputParser but for models/use-cases that do better with XML
# tags than JSON braces. Needs `pip install defusedxml` (not installed in this
# project's venv, which is why this section is commented out).

# from langchain_core.output_parsers import XMLOutputParser
#
# xml_parser = XMLOutputParser(tags=["country", "name", "capital"])
# xml_prompt = PromptTemplate(
#     template="Tell me the name and capital of France.\n{format_instructions}",
#     input_variables=[],
#     partial_variables={"format_instructions": xml_parser.get_format_instructions()},
# )
# xml_chain = xml_prompt | model | xml_parser
# print(xml_chain.invoke({}))
# -> {'country': [{'name': 'France'}, {'capital': 'Paris'}]}


# =============================================================================
# 6) Other list-shaped parsers
# =============================================================================
# MarkdownListOutputParser  -- parses a markdown bullet list ("- item") into list[str]
# NumberedListOutputParser  -- parses a numbered list ("1. item") into list[str]
# Same usage pattern as CommaSeparatedListOutputParser: get_format_instructions()
# -> put in the prompt, then parser.parse(text) / use it at the end of the chain.

# from langchain_core.output_parsers import MarkdownListOutputParser, NumberedListOutputParser


# =============================================================================
# 7) Tool-calling parsers  -- parse result.tool_calls, not result.content
# =============================================================================
# These are for when you've bound tools with model.bind_tools([...]) and want to
# pull the structured tool-call arguments out, rather than parsing free text.
# Unlike the parsers above, they read from the AIMessage's `tool_calls` field
# (see ../InvokeMetadata/metadata.py section 5), so they need no format
# instructions in the prompt at all -- the "formatting" is the provider's own
# function-calling API.
#
# JsonOutputToolsParser     -- returns list[{"type": tool_name, "args": {...}}]
# JsonOutputKeyToolsParser  -- like above but filtered to one tool by name, and
#                              can return just that tool's args (first_tool_only=True)
# PydanticToolsParser       -- same idea, but returns validated Pydantic instances
#                              (pass pydantic_schemas=[YourModel, ...])
#
# Example shape (not run here -- needs model.bind_tools(...) set up first):
#
# from langchain_core.output_parsers import PydanticToolsParser
# model_with_tools = model.bind_tools([Country])
# chain = model_with_tools | PydanticToolsParser(tools=[Country])
# chain.invoke("Tell me about France")  # -> [Country(name=..., capital=..., population=...)]


# =============================================================================
# 8) Not installed in this project, but worth knowing about
# =============================================================================
# The separate `langchain` package (pip install langchain, on top of
# langchain-core which this project already has) adds a few more parsers:
#   StructuredOutputParser  -- older, pre-Pydantic way to define a response schema
#   EnumOutputParser        -- restrict output to one value from a fixed Enum
#   DatetimeOutputParser    -- parses a datetime string into a Python datetime
#   OutputFixingParser      -- wraps another parser; on a parse failure, sends the
#                              broken output + the error back to the model to fix
#   RetryOutputParser       -- like OutputFixingParser, but re-runs with the
#                              original prompt instead of just the broken output
# These matter mainly for making the free-text-based parsers above (sections
# 2-6) more robust against models that don't follow formatting instructions
# reliably -- PydanticOutputParser + OutputFixingParser is a common pairing.


# =============================================================================
# Quick decision guide
# =============================================================================
# Just want the plain text answer                 -> StrOutputParser               (section 1)
# Quick list of items                             -> CommaSeparatedListOutputParser (section 2)
# Loose/dynamic JSON, no strict validation needed -> JsonOutputParser               (section 3)
# Need validation + a typed Python object back    -> PydanticOutputParser           (section 4)
# Model does better with XML than JSON            -> XMLOutputParser                (section 5)
# Model already supports function/tool calling    -> skip all of this, use
#                                                     model.bind_tools() + a tool parser (section 7)
#                                                     or with_structured_output() instead
