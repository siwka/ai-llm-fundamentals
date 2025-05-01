import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[{
        "role": "user",
        "content": "Why week has 7 days?"
    }]
)

print(response["message"]["content"])


response = ollama.chat(
    model="llama3.2",
    messages=[{
        "role": "user",
        "content": "What are days of the week?"
    }],
    stream=True
)

for chunk in response:
    print(chunk["message"]["content"], end=" ", flush=True)

    
print(ollama.show("llama3.2"))


# Create a new model with modelfile
nurse_role = "You are very smart nurse who knows everything about medications. You are very succinct and informative."

ollama.create(model="knowitall",
              from_="llama3.2",
              system=nurse_role)

response = ollama.generate(model="knowitall", prompt="What is the best medicine for migraine?")
print(response["response"])

# Delete a model
ollama.delete("knowitall")