from anthropic import Anthropic

anthropic = Anthropic()

print("Claude CLI REPL started. Type 'exit' to quit.")

while True:
    prompt = input("\nYou: ")
    if prompt.lower() in {"exit", "quit"}:
        break

    completion = anthropic.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=300,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    print("\nClaude:", completion.content[0].text)
