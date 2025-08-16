




# CLI TEST
# if __name__ == "__main__":
#     url = input("Enter article URL: ").strip()
#     text = fetch_article_text(url)

#     if not text:
#         print("❌ No text found.")
#     else:
#         print("\n--- Summary ---\n")
#         print(summarize_text(text, style="concise"))

#         print("\n--- Splitting into Chunks ---")
#         chunks = split_text(text)
#         print(f"✅ Created {len(chunks)} chunks. First chunk:\n", chunks[0])
 
#         print("\n--- Storing chunks in FAISS ---")
#         faiss_store = store_chunks_in_faiss(chunks, embedding_model, faiss_index_path)
#         print("✅ Chunks stored in FAISS. Index ready for retrieval.")

#         # Q&A Test
#         while True:
#             question = input("\nAsk a question about the article (or 'quit'): ").strip()
#             if question.lower() == "quit":
#                 break
#             answer = answer_question(question, faiss_store, client)
#             print("\n🤖 Answer:", answer)


# Test
# sample_text = """
# Artificial intelligence (AI) refers to the simulation of human intelligence in machines 
# that are programmed to think like humans and mimic their actions. 
# AI has applications in many industries including healthcare, finance, education, 
# and transportation. One of the key challenges in AI is ensuring ethical use.
# """

# # Test with different styles
# print("Concise Summary:\n", summarize_text(sample_text, style="concise"))
# print("\nDetailed Summary:\n", summarize_text(sample_text, style="detailed"))

