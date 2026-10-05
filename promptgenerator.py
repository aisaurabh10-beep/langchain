from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""You are an expert research assistant who explains academic papers clearly.

Paper:
{paper_text}

Audience: {audience}
Explanation style: {style}

Instructions:
- Summarize the core problem the paper addresses, the method/approach used, and the key findings.
- Tailor the vocabulary and depth to the specified audience.
- Follow the requested explanation style (e.g., concise summary, bullet points, detailed walkthrough, ELI5).
- Call out any notable limitations or open questions the paper mentions.
""",
    input_variables=["paper_text", "audience", "style"],
)

# Example usage:
# prompt = template.invoke({
#     "paper_text": "<paste abstract or full paper text here>",
#     "audience": "undergraduate CS student",
#     "style": "bullet-point summary",
# })
# result = model.invoke(prompt)
# print(result.content)


template.save("academic_paper_explainer_prompt.json")