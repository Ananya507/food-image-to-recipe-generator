import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Function to extract features from the food image using ResNet-50
def extract_features(img):
    model = models.resnet50(pretrained=True)  # Load ResNet-50 with ImageNet weights
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(img)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features

# Function to generate a recipe based on extracted features using GPT-2
def generate_recipe(prompt):
    model_name = 'gpt2-medium'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    input_text = prompt  # Use the provided prompt
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1000,  # Adjust max_length to generate longer recipes if needed
            temperature=1.0,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recipe

# Main function to perform image-to-recipe generation
def image_to_recipe(img, prompt):
    try:
        # Extract features from the food image
        features = extract_features(img)

        # Generate recipe based on the extracted features and prompt
        generated_recipe = generate_recipe(prompt)

        return generated_recipe
    except Exception as e:
        return f"Error generating recipe: {e}"

# Streamlit app
def main():
    st.title('Food Image to Recipe Generator')

    # Prompt input
    prompt = st.text_input('Enter recipe prompt:', 'Enter your prompt:')  # Default prompt

    # File uploader
    uploaded_file = st.file_uploader("Upload a food image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded food image.', use_column_width=True)

        # Convert image to PIL format
        img = Image.open(uploaded_file)

        # Button to generate recipe
        if st.button('Generate Recipe'):
            # Generate recipe
            generated_recipe = image_to_recipe(img, prompt)

            # Display generated recipe
            if isinstance(generated_recipe, str):
                st.subheader('Generated Recipe:')
                st.write(generated_recipe)
            else:
                st.subheader('Failed to generate recipe.')

if __name__ == '__main__':
    main()
