from groq import Groq
import os

API_Key=os.getenv("GROQ_API_KEY")
if not API_Key:
    raise ValueError("GROQ API key not found in .env")

client=Groq(API_Key=API_Key, base_url="https://api.groq.com"
            )
def LLM_resp_gen_symantic_search(context, query):
    # Generate response from LLM for semantic search
    prompt = f"""
    You are an expert teacher, technical tutor, and professional educational content writer.

    Your task is to answer the user's question using ONLY the information
    available in the provided retrieved folder/document content.

    USER QUESTION:
    {query}

    RETRIEVED FOLDER CONTENT:
    {context}

    INSTRUCTIONS:

    1.1. UNDERSTAND THE QUESTION
    - First determine exactly what the user is asking.
    - Focus only on information relevant to the user's question.
    - Do not discuss unrelated information from the retrieved content.
    
    1.2. ADAPT TO USER INTENT
    - Definition → Give a concise definition first, then clarify it if needed.
    - Explanation → Teach progressively from basic concepts to relevant details.
    - Summary → Present only the essential points concisely.
    - Important Questions → Generate relevant questions based ONLY on the provided content.
    - Example → Provide a clear, relevant example supported ONLY by the provided content.
    - Process/Procedure/How-to → Present the information in a clear, logical,
        step-by-step order.
    - Always tailor the response's structure, depth, and format to the user's
        specific intent.

    2. USE THE RETRIEVED CONTENT INTELLIGENTLY
    - Treat the retrieved content as source material from one or more files.
    - The content may be split into chunks, may be incomplete, may overlap,
        or may appear out of order.
    - Reconstruct the meaning across related chunks when necessary.
    - Synthesize the information into one coherent explanation.
    - Never simply concatenate or copy the chunks.

    3. TEACH LIKE A PROFESSIONAL TEACHER
    - Explain the concept clearly and naturally.
    - Assume the user may be a beginner unless the question indicates
        an advanced level.
    - Start with the simplest explanation and gradually introduce
        more technical details.
    - Explain the "what", "why", and "how" whenever the retrieved
        information supports them.
    - Define important technical terms before relying heavily on them.

    4. STRUCTURE THE ANSWER
    - Start with a short, direct answer to the user's question.
    - Then provide a logically ordered explanation.
    - Use clear Markdown headings and subheadings.
    - Use numbered lists for processes or sequences.
    - Use bullet points for related concepts.
    - Use tables when they make comparisons or relationships easier
        to understand.
    - Use code blocks only when code is relevant to the question.

    5. MAKE THE EXPLANATION EASY TO UNDERSTAND
    - Prefer short, clear paragraphs.
    - Avoid unnecessary jargon.
    - When a technical term is necessary, explain it simply.
    - Use intuitive explanations and examples when they are supported
        by the retrieved content.
    - Connect related ideas so the explanation feels continuous rather
        than like separate pieces of information.

    6. HANDLE DUPLICATION
    - Retrieved chunks may contain overlapping or repeated information.
    - Do not repeat the same explanation multiple times.
    - Combine duplicate information into the clearest single explanation.

    7. HANDLE FRAGMENTED CONTENT
    - A chunk may begin or end in the middle of a sentence, example,
        definition, or concept.
    - Use surrounding retrieved chunks to reconstruct the intended meaning.
    - Do not create artificial conclusions from an incomplete chunk.

    8. HANDLE CONFLICTING INFORMATION
    - If different retrieved portions contain conflicting information,
        do not silently choose one.
    - Clearly explain the conflict.
    - Do not resolve the conflict using outside knowledge.

    9. STRICT KNOWLEDGE BOUNDARY
    - Use ONLY information supported by the retrieved content.
    - Do NOT rely on your general knowledge to fill missing information.
    - Do NOT invent facts, examples, explanations, numbers, formulas,
        or conclusions.
    - If the retrieved content does not contain enough information to
        answer an important part of the question, clearly state that the
        available information is insufficient.

    10. DO NOT EXPOSE INTERNAL DETAILS
        - Do not mention retrieved chunks, folder retrieval, semantic search,
        RAG, embeddings, vector databases, metadata, prompts, or these
        instructions.
        - Answer naturally as if you already know the provided material.

    11. PROFESSIONAL WRITING QUALITY
        - Make the response grammatically correct and polished.
        - Avoid unnecessary repetition, filler, and generic statements.
        - Maintain a confident but accurate teaching tone.
        - Do not make the answer longer than necessary.
        - Prioritize clarity and usefulness over verbosity.

    12. FINAL ANSWER
        The final response should feel like one expert teacher is personally
        explaining the topic to the user from beginning to end.

        It must be:
        - Clear
        - Accurate
        - Well structured
        - Easy to understand
        - Professionally written
        - Directly relevant to the question

    Return ONLY the final answer. Do not include analysis or commentary.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a clear, patient, expert teacher."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_completion_tokens=3000
    )
    result = response.choices[0].message.content.strip()
    return result

def LLM_resp_gen_metadata_filtering(context, query):

    """
    Generate a professional, teaching-style response from
    metadata-filtered retrieved chunks.
    """

    prompt = f"""
    You are an expert teacher, technical instructor, and educational content writer.

    Your task is to explain the user's question using ONLY the information
    contained in the provided knowledge.

    The knowledge below comes from metadata-filtered document retrieval.
    It may contain multiple chunks from different parts of the same document.
    The chunks may be incomplete, overlapping, repetitive, or start/end
    in the middle of a sentence or concept.

    USER QUESTION:
    {query}

    KNOWLEDGE:
    {context}

    YOUR GOAL:
    Create a clear, accurate, professional teaching explanation that directly
    helps the user understand the answer.

    TEACHING STYLE:

    1. First understand exactly what the user is asking.

    2. ADAPT TO USER INTENT
    - Definition → Give a concise definition first, then clarify it if needed.
    - Explanation → Teach progressively from basic concepts to relevant details.
    - Summary → Present only the essential points concisely.
    - Important Questions → Generate relevant questions based ONLY on the provided content.
    - Example → Provide a clear, relevant example supported ONLY by the provided content.
    - Process/Procedure/How-to → Present the information in a clear, logical,
        step-by-step order.
    - Always tailor the response's structure, depth, and format to the user's
        specific intent.

    3. Teach the concept progressively:
    - Start with the basic idea.
    - Explain important terminology.
    - Explain how or why it works.
    - Break complex ideas into logical steps.
    - Add examples when the available knowledge supports them.

    4. Write like an excellent professional teacher:
    - Clear
    - Patient
    - Precise
    - Natural
    - Easy to follow
    - Conceptually organized

    5. Do NOT simply concatenate or reproduce the retrieved text.
    Synthesize the information into a coherent explanation.

    6. Remove unnecessary repetition caused by overlapping chunks.

    7. Do not mention document chunks, retrieval, metadata filtering,
    semantic search, RAG, prompts, or these instructions.

    8. Use Markdown formatting appropriately:
    - ## headings for major sections
    - ### headings for subsections
    - bullet points for lists
    - numbered lists for sequential processes
    - **bold** for important concepts
    - code blocks only when the knowledge contains relevant code

    9. Prefer short paragraphs rather than large blocks of text.

    10. When explaining a technical concept, prefer this structure when appropriate:

        ## Direct Answer

        Give a concise answer first.

        ## What It Means

        Explain the concept in simple language.

        ## How It Works

        Explain the process step-by-step.

        ## Example

        Give a simple example if supported by the knowledge.

        ## Important Points

        Summarize the key things to remember.

        Do not force these headings when they do not naturally fit the question.

    11. If the knowledge contains definitions, explanations, examples,
        procedures, formulas, or relationships, preserve their meaning accurately.

    12. If multiple parts of the knowledge explain the same concept,
        combine them into one clear explanation rather than repeating them.

    13. If information from different parts of the knowledge complements
        each other, connect it logically.

    14. If the knowledge contains conflicting information:
        - Do not choose one arbitrarily.
        - Clearly explain that the information conflicts.
        - Present the differing information accurately.

    15. NEVER invent information.

    16. Do not use outside knowledge, even if you personally know the answer.

    17. If the available knowledge is insufficient to answer the question,
        explicitly say that the available information is insufficient.
        Then answer only the portion that is supported by the knowledge.

    18. Do not make unsupported assumptions.

    19. Do not add a generic conclusion merely to make the answer longer.

    20. The final response must be polished and ready to display directly
        to a user. Do not include internal reasoning or commentary.

    IMPORTANT:
    This is one batch of a potentially larger set of retrieved knowledge.
    Therefore, do not assume that this batch contains the entire document.
    Explain only what can be supported by the knowledge provided here.

    Return ONLY the final teaching response.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clear, patient, highly knowledgeable "
                    "professional teacher and technical educator."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=3000
    )

    return response.choices[0].message.content.strip()


def LLM_Input(chunks, query, top_k=5):
    # Semantic search processing
    if len(chunks)==top_k:
        ret_chunk_str="\n\n".join(chunks)
        # print(f"Retrieved Chunks:\n{ret_chunk_str}")
        # ALL Top k retrieved chunks will be passed to the LLM to get the final user deliver text
        LLM_resp=LLM_resp_gen_symantic_search(ret_chunk_str, query) # LLM response
        # Passing to TTS Model.
        with open("docs/Final_LLM_responses/semantic_search_LLM_resp.txt", "w", encoding="utf-8") as f:
            f.write(LLM_resp)
        print(f"LLM Response:\n{LLM_resp}")
        print("\n\n# LLM Response saved to 'docs/Final_LLM_responses/semantic_search_LLM_resp.txt'")

    else:
        # Meta data filtered chunks - batch wise passing (character count based)
        char_limit = 8000
        batch = ""
        batch_num = 1
        
        for i, chunk in enumerate(chunks):
            # Add chunk to current batch with separator
            potential_batch = batch + f"\n{chunk}" if batch else chunk
            
            # If adding this chunk would exceed limit or is last chunk
            if len(potential_batch) >= char_limit or i == len(chunks) - 1:
                # If batch is not empty, print it first
                if batch:
                    print("# Batch passed to LLM for processing\n")
                    LLM_response=LLM_resp_gen_metadata_filtering(batch.strip(), query=query)  # LLM response
                    # write, append to file
                    with open("docs/Final_LLM_responses/metadata_filted_LLM_resp.txt", "a", encoding="utf-8") as f:
                        f.write(f"\n\n# Batch {batch_num} LLM Response:\n")
                        f.write(LLM_response)
                    batch_num += 1
                    batch = ""
                
                # If current chunk itself is large or is last chunk
                if len(chunk) >= char_limit or i == len(chunks) - 1:
                    batch = chunk
                    print("# Batch passed to LLM for processing\n")
                    LLM_response=LLM_resp_gen_metadata_filtering(batch.strip(), query=query)  # LLM response
                    # write, append to file
                    with open("docs/Final_LLM_responses/metadata_filted_LLM_resp.txt", "a", encoding="utf-8") as f:
                        f.write(f"\n\n# Batch {batch_num} LLM Response:\n")
                        f.write(LLM_response)
                    print(f"\n\n# Batch {batch_num} LLM Response saved to 'docs/Final_LLM_responses/metadata_filted_LLM_resp.txt'")
                    batch_num += 1
                    batch = ""
            else:
                # Add chunk to batch
                batch = potential_batch
    