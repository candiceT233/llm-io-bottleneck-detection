import os
from dotenv import load_dotenv
from openai import OpenAI
import yaml

# Load env variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt(snapshot: dict) -> str:

    snapshot_text = yaml.safe_dump(snapshot, sort_keys=False)

    return f"""
You are an expert in HPC I/O performance.

Analyze the following workflow execution snapshot and identify the I/O bottleneck.

Possible categories:
- storage_contention
- small_file_overhead
- producer_consumer_mismatch
- storage_tier_mismatch
- serialization_bottleneck

Return your answer in this format:
Bottleneck: <category>
Explanation: <really short reasoning>

Snapshot:
{snapshot_text}
"""


def diagnose(snapshot: str, model: str = "gpt-4.1-mini") -> str:
    prompt = build_prompt(snapshot)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an HPC systems expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# Example usage
def load_snapshot_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    snapshot = load_snapshot_yaml("workflow_snapshots/ex01.yaml")

    result = diagnose(snapshot)

    print("\n=== Model Diagnosis ===")
    print(result)

    print("\n=== Ground Truth ===")
    print(snapshot["ground_truth"]["bottleneck"])