# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

CLIP_FRIENDLY = """
    The following are the result of captioning two groups of images:

    {text}

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not talk about the caption, e.g., "caption with one word" and do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images"
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"

    Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
"""

VLM_PROMPT = {"no_context": """
    This image contains two groups of images. 20 images from Group A are shown in the first two rows, while 20 images from Group B are shown in the last two rows.

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images"
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"

    Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images in group A compared to group B. Answer with a list (separated by bullet points "*"). Your response:
""",
    "centered": """
    This image contains two groups of images. 20 images from Group A are shown in the first two rows, while 20 images from Group B are shown in the last two rows.

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images"
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"
    Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images in group A compared to group B. The object of interest is centered in each image. Ignore blacked-out regions. Answer with a list (separated by bullet points "*"). Your response:
""",
    "red_bbox": """
    This image contains two groups of images. 20 images from Group A are shown in the first two rows, while 20 images from Group B are shown in the last two rows.

    I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images"
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"
    Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images in group A compared to group B. The object of interest is highlighted with a red bounding box in each image. Answer with a list (separated by bullet points "*"). Your response:
"""}


VLM_SYSTEM_PROMPT = """
    You are an expert data scientist helping another machine learning researcher analyze two groups of images (Group A and Group B). Your task is to identify and summarize the major differences between these two groups of images based on their content.
    Provide concise and clear captions that capture the key concepts that distinguish Group A from Group B.

    Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"
    * "a joyful atmosphere"

    Do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images"
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"
"""

VLM_USER_PROMPT_GROUP_1 = """
    These are the images from Group A:
"""
VLM_USER_PROMPT_GROUP_2 = """
    These are the images from Group B:
"""
VLM_USER_PROMPT_RESPONSE = {"no_context": """
    Again, based on the two groups of images (Group A and Group B) from the above, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images in Group A compared to Group B. Answer with a list (separated by bullet points "*").
    Your response:
""",
    "centered": """
    Again, based on the two groups of images (Group A and Group B) from the above, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images in Group A compared to Group B. The object of interest is centered in each image. Ignore blacked-out regions. Answer with a list (separated by bullet points "*").
    Your response:
""",
    "red_bbox": """
    Again, based on the two groups of images (Group A and Group B) from the above, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images in Group A compared to Group B. The object of interest is highlighted with a red bounding box in each image. Answer with a list (separated by bullet points "*").
    Your response:
"""}

VLM_SYSTEM_PROMPT_ZEROSHOT = """
    You are an expert data scientist helping another machine learning researcher analyze two groups of images (Group A and Group B). Your task is to identify and summarize the major differences between these two groups of images based on their content.
    Provide concise and clear captions that capture the key concepts that distinguish Group A from Group B.

    Come up with 5 distinct concepts that are more likely to be true for Group A compared to Group B. Sort the concepts in order of importance, with the most distinctive differences listed first.
    Please write a list of captions (separated by bullet points "*"). For example:
    * "a dog next to a horse"
    * "a car in the rain"
    * "low quality"
    * "cars from a side view"
    * "people in a intricate dress"

    Do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
    * INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
    * INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
    * INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
    * INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
    * INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
    * INCORRECT: "More realistic images" CORRECTED: "realistic images"
    * INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"
"""

CAPTIONING_PROMPT = {
    "no_context": "Describe this image in detail.",
    "centered": "Describe the object in the center of this image in detail. Ignore blacked-out areas.",
    "red_bbox": "Describe the object inside the red bounding box in this image in detail.",
}

FEATURE_PROMPT = "Describe this image in 1-3 words."
