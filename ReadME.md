### Feeling Your Images: Visual Emotion Recognition based on Image Attributes.

### Objective:
Our primary goal is to move beyond conventional approaches that predominantly focus on facial features.
Instead, we aim to develop a robust system that comprehensively understands the visual context of an entire image, offering a nuanced interpretation of the scene.

### Background:
Computer vision, a subfield of artificial intelligence, has witnessed remarkable progress in recent years.
However, existing models often fall short when it comes to understanding the broader context of visual information.
Our project addresses this limitation by leveraging the capabilities of Transformer classifiers.

### Transformer Classifiers:
The Transformer architecture, originally proposed for natural language processing, has demonstrated unparalleled success in capturing long-range dependencies.
We apply this architecture to the domain of computer vision, employing self-attention mechanisms to enable our model to consider global relationships within an image.

### Key Features:

    1. Holistic Image Understanding: Our model surpasses traditional facial-centric approaches, considering the entire image for a more comprehensive analysis.

    2. Adaptability: The Transformer's self-attention mechanism allows our model to adapt dynamically to different visual contexts, enhancing its versatility across various scenes.

    3. Robustness: By considering the entire image, our model proves to be more resilient to occlusions, variations in lighting, and other challenges that may confound traditional classifiers.

### Methodology:

    0. Dataset and Pretrained model: We used as based  EmoSet (https://vcc.tech/EmoSet) with images labeled for feelings (based on Plutchik’s Wheel of Emotions) to fine tunning the pretrained model made by Google (https://huggingface.co/google/vit-base-patch16-224) as base of our project.
    
    1. Data Processing: We carefully process the dataset to fits on model requirements, based on the research. 
    
    2. Train and hyper parameter tunning: We train our model. We choose the hyper parameters based on Population-based Training algorithm (ray tune python library). 
    
    3. Evalutation: It's evaluate the final results using reserved part of the dataset. The two used metrics are a confusion matrix and accuracy. 
    
    4. The visual final product: A movie with captions for the predict labels.