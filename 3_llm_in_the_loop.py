import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


# Predefined refinement goals
refinement_goals = [
    "relieving stress",
    "promoting physical wellness",
    "improving productivity",
    "enhancing mental clarity",
    "providing emotional support"
]

print("Starting Loading model..")

# Load once at the module level for efficiency
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # or torch.float16 if your hardware supports it
    device_map="auto"
)

print("Model Loaded!!")

# LLM call function (replace with your actual LLM API call)
# def call_llm(prompt, model="llama-3.1-8B-Instruct"):
#     # Example using OpenAI-like API
#     # response = openai.ChatCompletion.create(
#     #     model=model,
#     #     messages=[{"role": "user", "content": prompt}]
#     # )
#     # return response['choices'][0]['message']['content']
#     # Replace the above with your actual LLM call
#     raise NotImplementedError("Replace this with your LLM API call")

def call_llm(prompt, system_prompt=None, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """
    Call Llama-3.1-8B-Instruct with a user/system prompt and return the generated text.

    Args:
        prompt (str): The user message.
        system_prompt (str, optional): The system message for context.
        max_new_tokens (int): Max tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.

    Returns:
        str: The model's generated response.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Use return_dict=True to get both input_ids and attention_mask
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    # Move tensors to model device
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    # Set pad_token_id to eos_token_id as recommended
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )

    response = outputs[0, input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True).strip()

# 1. Analyze user summary and infer refinement goals
def get_refinement_goals(user_summary, refinement_goals):
    # Define the mapping from features to categories
    activity_features = [
        "calories", "steps", "distance", "lightly_active_minutes",
        "moderately_active_minutes", "very_active_minutes", "sedentary_minutes"
    ]
    sleep_features = ["sleep_duration", "sleep_efficiency"]
    stress_features = [
        "stress_score", "rmssd", "nremhr", "resting_hr", "bpm", "mindfulness_session"
    ]

    # Create a mapping for the prompt
    feature_category_map = {}
    for feature in user_summary:
        if feature in activity_features:
            feature_category_map[feature] = "Activity Level"
        elif feature in sleep_features:
            feature_category_map[feature] = "Sleep Quality"
        elif feature in stress_features:
            feature_category_map[feature] = "Stress Level"
        else:
            feature_category_map[feature] = "Other"

    # Build the prompt for the LLM
    cot_prompt = f"""
You are an expert assistant. Given the following user summary (with activity, sleep, and stress trends), and a list of possible refinement goals, analyze the summary and select the most relevant goals for this user. For your analysis, use the provided mapping of each feature to its category (Activity Level, Sleep Quality, Stress Level). Output only a comma-separated list of the most relevant goals.

User Summary (feature: trend):
{json.dumps(user_summary, indent=2)}

Feature Category Mapping:
{json.dumps(feature_category_map, indent=2)}

Refinement Goals: {', '.join(refinement_goals)}
"""

    # Uncomment below and replace with actual LLM call
    goals_str = call_llm(cot_prompt)
    # For demonstration, hardcode the answer:
    # goals_str = "relieving stress, enhancing mental clarity"
    selected_goals = [g.strip() for g in goals_str.split(",") if g.strip()]
    return selected_goals


# 2. Rewrite the user query to include the selected refinement goals
def rewrite_query(user_query, selected_goals):
    rewrite_prompt = f"""
You are an AI assistant that rewrites user queries based on their context and refinement goals.
Original Query: "{user_query}"
Refinement Goals: {', '.join(selected_goals)}
Rewrite the query to explicitly include one or more of these goals, making it concise and contextually relevant.
Output only the rewritten query.
"""
    # Uncomment below and replace with actual LLM call
    rewritten_query = call_llm(rewrite_prompt)
    # For demonstration, hardcode the answer:
    # rewritten_query = "How can I prepare for my presentation tomorrow to relieve stress and enhance mental clarity?"
    return rewritten_query

# 3. Generate the solution to the rewritten query
def generate_solution(rewritten_query):
    solution_prompt = f"""
You are an expert assistant. Provide a practical, evidence-based answer to the following query, focusing on actionable steps and clear reasoning.

Query: {rewritten_query}
"""
    # Uncomment below and replace with actual LLM call
    solution = call_llm(solution_prompt)
    # For demonstration, hardcode the answer:
    # solution = (
    #     "To prepare for your presentation while relieving stress and enhancing mental clarity, try the following: "
    #     "1. Review your key points the night before and again in the morning. "
    #     "2. Practice deep breathing or mindfulness for 5-10 minutes. "
    #     "3. Get a good night's sleep. "
    #     "4. Take a brisk walk or do light exercise to boost focus. "
    #     "5. Visualize your success and remind yourself of your strengths."
    # )
    return solution

# Main workflow
def main():
    print("Paste your compact JSON user summary (single line), then press Enter:")
    summary_str = input().strip()
    try:
        user_summary = json.loads(summary_str)
    except Exception as e:
        print("Invalid JSON. Please check your input.")
        return

    user_query = input("Enter the user query: ").strip()

    selected_goals = get_refinement_goals(user_summary, refinement_goals)
    rewritten_query = rewrite_query(user_query, selected_goals)
    solution = generate_solution(rewritten_query)

    print("\nSelected Refinement Goals:", selected_goals)
    print("Rewritten Query:", rewritten_query)
    print("Solution:", solution)

if __name__ == "__main__":
    main()
