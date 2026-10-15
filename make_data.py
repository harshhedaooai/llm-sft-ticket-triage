"""Generate a synthetic support-ticket dataset in OpenAI chat-style JSONL (the same
format the OpenAI SFT API uses). Label = category + priority."""
import json, random
random.seed(42)

SYSTEM = "You are a support-ticket triage assistant. Reply only with JSON."
PLANS = ["Pro", "Basic", "Premium", "Student", "Team"]
ITEMS = ["laptop", "headphones", "phone case", "keyboard", "backpack", "monitor"]

BODIES = {
 "billing": ["I was charged twice for my {p} subscription this month.",
             "My invoice shows the wrong amount for the {p} plan.",
             "Why was my card billed after I cancelled the {p} plan?",
             "I need a GST invoice for my {p} payment."],
 "technical": ["The app crashes every time I open the {p} dashboard.",
               "I get error 500 when I upload a file on the {p} plan.",
               "The website is very slow and the page keeps freezing.",
               "Login works but the {p} reports page shows a blank screen."],
 "account": ["I forgot my password and the reset email never arrives.",
             "Please change the email address on my {p} account.",
             "My account got locked after too many login attempts.",
             "How do I delete my {p} account permanently?"],
 "shipping": ["My {i} order has not arrived and tracking is stuck.",
              "The courier marked my {i} as delivered but I never got it.",
              "Can I change the delivery address for my {i} order?",
              "My {i} shipment is delayed by over a week."],
 "refund": ["I want a refund for the {i} I bought, it arrived damaged.",
            "The {i} does not work. Please refund my money.",
            "I returned my {i} two weeks ago and still have no refund.",
            "Please cancel my order and refund the {i} payment."],
}
SUFFIX = {
 "high": ["This is urgent, please fix it immediately.", "I am very angry, respond today!",
          "Critical issue, my business is blocked."],
 "medium": ["Please look into it.", "Kindly help me with this.", ""],
 "low": ["No rush, just letting you know.", "Whenever you get time is fine.",
         "Just a small question."],
}
GREET = ["Hi,", "Hello team,", "Dear support,", ""]

def make_example():
    cat = random.choice(list(BODIES)); pri = random.choice(list(SUFFIX))
    body = random.choice(BODIES[cat]).format(p=random.choice(PLANS), i=random.choice(ITEMS))
    text = " ".join(x for x in [random.choice(GREET), body, random.choice(SUFFIX[pri])] if x)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
        {"role": "assistant", "content": json.dumps({"category": cat, "priority": pri})}]}

def main(n=600):
    seen, rows = set(), []
    while len(rows) < n:
        ex = make_example(); key = ex["messages"][1]["content"]
        if key not in seen:
            seen.add(key); rows.append(ex)
    splits = {"train": rows[:int(.8*n)], "val": rows[int(.8*n):int(.9*n)], "test": rows[int(.9*n):]}
    for name, data in splits.items():
        with open(f"data/{name}.jsonl", "w") as f:
            for r in data: f.write(json.dumps(r) + "\n")
        print(name, len(data))

if __name__ == "__main__":
    main()
