from datasets import load_dataset
from itertools import islice
from pathlib import Path
import shutil

RAW_DIR = Path("data/raw")

if RAW_DIR.exists():
    shutil.rmtree(RAW_DIR)
RAW_DIR.mkdir(parents=True, exist_ok=True)

PERCENTAGE = 0.05
TOTAL_ROWS = 20000
MAX_ROWS = int(TOTAL_ROWS * PERCENTAGE)

dataset = load_dataset(
    "camel-ai/physics",
    split="train",
    streaming=True
)

dataset = islice(dataset, MAX_ROWS)

chunks_per_file = 300
file_index = 0
chunk_count = 0
current_file_chunks = []

for row in dataset:
    content = row.get("message_2", "").strip()
    if not content:
        continue

    topic = row.get("topic;", "")
    sub_topic = row.get("sub_topic", "")

    chunk_text = (
        f"Topic: {topic}\n"
        f"Subtopic: {sub_topic}\n\n"
        f"{content}"
    )

    current_file_chunks.append(chunk_text)
    chunk_count += 1

    if chunk_count % chunks_per_file == 0:
        file_path = RAW_DIR / f"physics_chunks_{file_index}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(current_file_chunks))

        file_index += 1
        current_file_chunks = []

# flush
if current_file_chunks:
    file_path = RAW_DIR / f"physics_chunks_{file_index}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(current_file_chunks))

print(f"Processed {chunk_count} rows (streamed).")