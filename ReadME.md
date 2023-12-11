## Feeling Your Images: Visual Emotion Recognition based on Image Attributes.
### Matias Lessa Vaz

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

### Vision Transformer (ViT) Components:​

![Vision Transformer from Google Diagram](./vit_figure.png)

    1. Input Image:​

    The input to the model is a standard image (e.g., a photograph).​

    2. Patch + Position Embedding:​

    The image is divided into fixed-size patches (e.g., 16x16 pixels each).​

    These patches are then flattened into a one-dimensional vector (hence the term "flattened patches").​

    Each flattened patch is then passed through a linear projection to convert it into a size compatible with the transformer encoder. This is effectively a learned embedding that maps the pixel intensities to a higher-dimensional space.​

    Positional embeddings are added to the flattened patches to retain the positional information. Since the transformer architecture does not inherently process sequential data, unlike RNNs, positional embeddings are crucial for maintaining the order of the patches.​

    An extra learnable embedding, often referred to as the "class token" ([class]), is prepended to the sequence of embedded patches. The state of this class token at the output of the transformer encoder will serve as the representation of the image.​

    3. Transformer Encoder:​

    The sequence of embedded patches, along with the class token, is fed into the transformer encoder.​

    The transformer encoder consists of a stack of identical layers (Lx indicates L layers stacked). Each layer has two main components:​

        1. Multi-Head Attention: This module allows the model to weigh different parts of the image patches differently, effectively allowing the model to "attend" to certain parts of the image more when making predictions. The multi-head part means that the attention mechanism is applied in parallel multiple times with different learned weights, allowing the model to capture different types of dependencies in the data.​

        2. MLP (Multi-Layer Perceptron): After attention has been applied, the result goes through a feedforward neural network (MLP), which further processes the information.​

    Each of these components is followed by normalization (Norm), which helps in stabilizing the learning process. Additionally, there are residual connections (depicted by the + signs) around each of the main components, which help in avoiding the vanishing gradient problem by allowing gradients to flow through the network.​

    4. MLP Head:​

    The state of the class token at the output of the transformer encoder is then passed through an MLP head. This MLP head is typically a smaller network that maps the representation to the final output space.​

    For classification tasks, this MLP head will have as many outputs as there are classes, and it typically uses a softmax function to turn the logits into probabilities.​

    5. Class Output:​

    The final output is a probability distribution over the classes, indicating how likely the model thinks the image belongs to each of the given classes (e.g., Bird, Ball, Car, etc.).​

### About Labels

Below, Diagram of Plutchik’s Wheel of Emotions, the theory behind of the dataset labels.

![Plutchik’s Wheel of Emotions](./WheelOfEmotions.png)

Plutchik's Wheel of Emotions is a psychological model that represents various human emotions and their relationships. Developed by Robert Plutchik, a psychologist, the wheel organizes emotions into primary, secondary, and tertiary categories based on their intensity and combinations. The wheel consists of eight primary emotions, and each primary emotion has an opposite.

Here's a breakdown of the primary emotions in Plutchik's Wheel:

    Joy - Opposite: Sadness
    Trust - Opposite: Disgust
    Fear - Opposite: Anger
    Surprise - Opposite: Anticipation
    Sadness - Opposite: Joy
    Disgust - Opposite: Trust
    Anger - Opposite: Fear
    Anticipation - Opposite: Surprise

The wheel also includes the concept of blends, where adjacent emotions can combine to form secondary emotions. For example, combining joy and trust results in love, while combining fear and surprise leads to awe.

The model is useful for understanding the complexity of human emotions and how they relate to one another. Plutchik's Wheel of Emotions can be applied in various fields, including psychology, design, and human-computer interaction, to better grasp the emotional aspects of human experiences.


The dataset labels to our training with the respective percentage for each class.

![Dataset Labels](./labels.png)

Because of the low variance of percentage between classes and the high number of samples, we decide to don't resample to uniformize the number os labels. That, together with the final results showing is harder to classify the overrepresented classes (normally, what happens is on the other way), leads us to decide to stay with all samples.