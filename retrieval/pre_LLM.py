from run_retrieval import retrieve_type

retrieved_data=retrieve_type(
    "Ernst Renan, ‘What is a Nation",
    INP="metadata_filtering")
print("Retrieved data:", retrieved_data)


# page_content='The first clear expression of nationalism...'
# metadata={
#     'chapter_name': 'The Rise of Nationalism in Europe',
#     'section_name': 'The French Revolution and the Idea of the Nation',
#     'chunk_index': 1
# }