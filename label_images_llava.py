#!/usr/bin/env python3


"""
Script for labeling images with tags based on the image content using 
the LLaVA / BakLLaVA multimodal LLM model. As the local inference can be slow, 
the script is intended only for training data generation. The actual tagger 
will be much more efficient model, trained on the generated data (~ distilled).
"""


import os
import argparse
from urllib.request import urlretrieve
import base64
from tqdm import tqdm

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


parser = argparse.ArgumentParser(
    description="Label the images with tags based on the image content."
)
parser.add_argument(
    "images", 
    type=str, 
    help="The directory with the input images."
)
parser.add_argument(
    "label_file",
    type=str,
    help="The file with the labels - every line contains <image_name> <label1> <label2> ... <labelN>.",
)
parser.add_argument(
    "--model_dir",
    type=str,
    default=".",
    help="The directory with the LLaMA and CLIP model. The models will be downloaded if not present.",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="bakllava1-q4",
    choices=["llava15-7b-f16", "bakllava1-f16", "bakllava1-q4"],
    help="Multimodal model to use.",
)


label_questions = {
    "sunny": "Is it sunny in the image?",
    "cloudy": "Is it cloudy in the image?",
    "raining": "Is it raining in the image?",
    "snowing": "Is it snowing in the image?",
    "foggy": "Is it foggy in the image?",
    "ground_snow": "Is there snow on the ground in the image?",
    "ground_water": "Is there water on the ground in the image?",
    "ground_leaves": "Can you see fallen leaves on the ground in the image?",
    "tree_leaves": "Are there leaves on the trees in the image?",
}

common_specification = 'Reply only with "yes" or "no".'

img_exts = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")


def main(args):
    if args.model_type == "llava15-7b-f16":
        model_llama_name = "llava15-7b-f16.gguf"
        model_llama_url = "https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-f16.gguf?download=true"
        model_llama_path = os.path.join(args.model_dir, model_llama_name)

        model_clip_name = "llava15-7b-clip-f16.gguf"
        model_clip_url = "https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf?download=true"
        model_clip_path = os.path.join(args.model_dir, model_clip_name)

    elif args.model_type == "bakllava1-f16":
        model_llama_name = "bakllava1-f16.gguf"
        model_llama_url = "https://huggingface.co/mys/ggml_bakllava-1/resolve/main/ggml-model-f16.gguf?download=true"
        model_llama_path = os.path.join(args.model_dir, model_llama_name)

        model_clip_name = "bakllava1-clip-f16.gguf"
        model_clip_url = "https://huggingface.co/mys/ggml_bakllava-1/resolve/main/mmproj-model-f16.gguf?download=true"
        model_clip_path = os.path.join(args.model_dir, model_clip_name)

    elif args.model_type == "bakllava1-q4":
        model_llama_name = "bakllava1-q4.gguf"
        model_llama_url = "https://huggingface.co/mys/ggml_bakllava-1/resolve/main/ggml-model-q4_k.gguf?download=true"
        model_llama_path = os.path.join(args.model_dir, model_llama_name)

        model_clip_name = "bakllava1-clip-f16.gguf"
        model_clip_url = "https://huggingface.co/mys/ggml_bakllava-1/resolve/main/mmproj-model-f16.gguf?download=true"
        model_clip_path = os.path.join(args.model_dir, model_clip_name)

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    if not os.path.exists(model_llama_path):
        print(f"Downloading the LLaMA model to {model_llama_path}")
        urlretrieve(model_llama_url, model_llama_path)

    if not os.path.exists(model_clip_path):
        print(f"Downloading the CLIP model to {model_clip_path}")
        urlretrieve(model_clip_url, model_clip_path)

    chat_handler = Llava15ChatHandler(clip_model_path=model_clip_path)
    llm = Llama(
        model_path=model_llama_path,
        chat_handler=chat_handler,
        n_ctx=2048,  # n_ctx should be increased to accomodate the image embedding
        logits_all=True,  # needed to make llava work
        verbose=False,
    )

    image_list = [
        os.path.join(args.images, img_name)
        for img_name in os.listdir(args.images)
        if img_name.endswith(img_exts)
    ]
    f_label_file = open(args.label_file, "at")

    for img_path in tqdm(image_list):
        img_name = os.path.basename(img_path)
        img_base64 = encode_image(img_path)
        tags = []

        print("### " + img_name)

        for label, question in label_questions.items():
            response_dict = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": 'You are an assistant who can perfectly determine what is in the given image and replies only with "yes" or "no" depending on the question.',
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_base64}},
                            {"type": "text", "text": question + common_specification},
                        ],
                    },
                ]
            )

            response_str = response_dict["choices"][0]["message"]["content"]

            print("-------------------")
            print("question:")
            print(question)
            print("response:")
            print(response_str)

            words = response_str.strip().split(" ")
            if len(words) != 1:
                print("WARN: Unexpected response from the model: " + response_str)
                continue

            yn = words[0].strip().lower()

            if yn not in ["yes", "no"]:
                print("WARN: Unexpected response from the model: " + response_str)

            if yn == "yes":
                tags.append(label)

            print("label assigned:")
            print(yn)

        tags_str = " ".join(tags)
        f_label_file.write(f"{img_name} {tags_str}\n")

    f_label_file.close()


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
