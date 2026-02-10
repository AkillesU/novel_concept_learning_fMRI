import os
from datasets import load_dataset
from PIL import Image

# 1. SETUP: Define your API token and target settings
# Replace with your actual token from https://huggingface.co/settings/tokens
with open("images/token.txt","r") as f:
    HF_TOKEN = f.read()
# Define the categories you want (Label ID -> Category Name)
# You can find the full mapping of 1000 classes online (e.g., https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)
TARGET_CATEGORIES = {
    409: "analog clock",          # CSV: "(analog) clock"
    414: "backpack",
    448: "birdhouse",
    450: "bobsled",
    457: "bow tie",
    470: "candle",
    473: "can opener",
    482: "cassette player",
    504: "coffee mug",
    505: "coffeepot",
    520: "crib",
    530: "digital clock",
    535: "disk brake",
    545: "electric fan",
    550: "espresso maker",
    572: "goblet",
    587: "hammer",
    600: "hook",
    606: "iron",
    613: "joystick",
    618: "ladle",
    619: "lampshade",
    623: "letter opener",
    626: "lighter",
    630: "loafer",
    632: "loudspeaker",
    636: "mailbag",
    647: "measuring cup",
    665: "moped",
    669: "mosquito net",
    671: "mountain bike",
    675: "moving van",
    695: "padlock",
    697: "pajama",
    707: "pay-phone",
    711: "perfume",
    720: "pill bottle",
    754: "radio",
    755: "radio telescope",
    760: "refrigerator",
    767: "rubber eraser",
    770: "running shoe",
    792: "shovel",
    797: "sleeping bag",
    800: "slot (machine)",         # ImageNet label: "slot, one-armed bandit"
    804: "soap dispenser",
    813: "spatula",
    818: "spotlight",
    828: "strainer",
    840: "swab",
    841: "sweatshirt",
    842: "swimming trunks",
    846: "table lamp",
    849: "teapot",
    861: "toilet seat",
    862: "torch",
    868: "tray",
    878: "typewriter keyboard",
    879: "umbrella",
    883: "vase",
    893: "wallet",
    896: "washbasin",
    968: "cup",
}


IMAGES_PER_CATEGORY = 150
OUTPUT_DIR = "./images/imagenet_samples"

def download_imagenet_samples():
    print(f"Stream-loading ImageNet-1k... (This may take a moment to initialize)")
    
    # 2. STREAMING: Load the dataset in streaming mode
    # 'split="train"' gives you the main training set.
    # 'streaming=True' prevents downloading the whole dataset.
    dataset = load_dataset(
        "imagenet-1k", 
        split="train", 
        streaming=True, 
        token=HF_TOKEN,
        trust_remote_code=True
    )

    # Dictionary to keep track of how many we've saved per category
    saved_counts = {label_id: 0 for label_id in TARGET_CATEGORIES}
    
    # Create output directories
    for label_id, name in TARGET_CATEGORIES.items():
        os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)

    print("Iterating through stream...")
    
    # 3. ITERATION: Loop through the stream
    for i, sample in enumerate(dataset):
        # Stop if we have collected enough images for ALL target categories
        if all(count >= IMAGES_PER_CATEGORY for count in saved_counts.values()):
            print("All targets reached! Stopping download.")
            break

        label = sample['label']
        
        # Check if this image belongs to a category we want
        if label in TARGET_CATEGORIES:
            current_count = saved_counts[label]
            
            if current_count < IMAGES_PER_CATEGORY:
                category_name = TARGET_CATEGORIES[label]
                image = sample['image'] # This is a PIL Image object
                
                # Save the image
                filename = f"{category_name}_{current_count}.jpg"
                save_path = os.path.join(OUTPUT_DIR, category_name, filename)
                
                # Convert to RGB to ensure compatibility (some are grayscale)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    
                image.save(save_path)
                
                saved_counts[label] += 1
                print(f"Saved {category_name} ({saved_counts[label]}/{IMAGES_PER_CATEGORY})")

if __name__ == "__main__":
    download_imagenet_samples()