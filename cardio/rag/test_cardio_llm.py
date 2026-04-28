from cardio_llm import generate_cardio_answer

question = "What should a patient with high BP and high cholesterol do?"

answer = generate_cardio_answer(question)

print("Cardio LLM Answer")
print("-----------------")
print(answer)