import json
import openai
import os
from tqdm import tqdm
from datasets import load_dataset

# Load COCO Karpathy dataset
dataset = load_dataset("yerevann/coco-karpathy")
COCO_ROOT = "/home/Downloads/coco" 
COCO_TRAIN_DIR = os.path.join(COCO_ROOT, "train2014/")
COCO_VAL_DIR = os.path.join(COCO_ROOT, "val2014/")
COCO_TEST_DIR = COCO_VAL_DIR  # Karpathy test set is from val2014

def load_images_and_captions(split):
    """
    Load image file paths and corresponding captions for the given split.
    """
    split_data = dataset[split]

    image_paths = []
    captions = []

    for entry in split_data:
        if "train2014" in entry["filepath"]:
            img_path = os.path.join(COCO_TRAIN_DIR, entry["filename"])
        else:
            img_path = os.path.join(COCO_VAL_DIR, entry["filename"])  

        longest_caption = max(entry["sentences"], key=lambda s: len(s.split()))

        image_paths.append(img_path)
        captions.append(longest_caption)

    return image_paths, captions

# Load coco train
train_images, train_captions = load_images_and_captions("train")
#test_images, test_captions = load_images_and_captions("test")

def extract_entities_gpt(image_paths, captions):
    results = {}
    for img_path, caption in tqdm(zip(image_paths, captions), total=len(image_paths)):
        prompt = f"""
        Given an image caption, decompose the caption into an atomic entity. Each entity should preserve descriptive details (e.g., size, color, material, location) together with the entity in a natural, readable phrase. The entity should only contain a noun and reserve noun modifiers in the caption. Please ignore the entities like 'a photo of', ‘an image of’, 'an overhead shot', 'the window showing' that are invisible in the image and ignore the entities like 'one' and 'the other' that have duplicate entities before.
        
        Caption:
        two cars are traveling on the road and wait at the traffic light.
        Entity:
        cars, road, traffic light

        Caption:
        duplicate images of a girl with a blue tank top and black tennis skirt holding a tennis racqet and swinging at a ball.
        Enity:
        girl, blue tank top, black tennis skirt, tennis racqet, ball

        Caption:
        the window showing a traffic signal is covered in droplets of rainwater.
        Entity:
        traffic signal, droplets of rainwater

        Caption:
        an overhead shot captures an intersection with a "go colts" sign.
        Entity:
        intersection, "go colts" sign

        Caption:
        a van with a face painted on its hood driving through street in china.
        Entity:
        van, a face painted on its hood, street in china

        Caption: two men, one with a black shirt and the other with a white shirt, are kicking each other without making contact.
        Entity: men, black shirt, white shirt

        Caption:
        {caption}
        Entity:
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                      {"role": "user", "content": prompt}]
        )
        try:
            if response.choices and len(response.choices) > 0:
                extracted_entities = response.choices[0].message.content.strip()
                results[img_path] = [entity.strip() for entity in extracted_entities.split(",")]
            else:
                print(f"Image {img_path}: Response does not contain 'choices'. Skipping.")
        except Exception as e:
            print(f"Image {img_path}: Error occurred - {e}. Skipping.")
    
    return results

# Save extracted entities to a JSON file
def save_to_json(data, output_file):
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    openai.api_key = 'please enter your openai api key'
    output_json_path = 'please enter the output json path'
    extracted_entities = extract_entities_gpt(train_images, train_captions)
    save_to_json(extracted_entities, output_json_path)
    
    print(f"Extracted entities saved to {output_json_path}")
