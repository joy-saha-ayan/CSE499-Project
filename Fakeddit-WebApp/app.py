import streamlit as st
from Model import BERTResNetClassifier
import torch
from transformers import BertModel, AutoTokenizer
from torchvision.transforms import v2
from PIL import Image

# Load tokenizer and BERT only once
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)
bert_model.eval()


def get_bert_embedding(text):
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=80,
        truncation=True,
        padding="max_length",
    )

    return (
        inputs["input_ids"].squeeze(0),
        inputs["attention_mask"].squeeze(0),
    )


st.title("Fakeddit App")

uploaded_title = st.text_input("Article Headline", "Lorem Ipsum")
uploaded_file = st.file_uploader(
    "Choose an accompanying image for the article...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Uploaded Image Preview", width="stretch")

if st.button("Predict"):

    if uploaded_file is not None:

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform_func = v2.Compose([
            v2.Resize((256, 256)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ])

        input_ids, attention_mask = get_bert_embedding(uploaded_title)

        input_image = Image.open(uploaded_file).convert("RGB")
        img_tensor = transform_func(input_image).unsqueeze(0)

        model = BERTResNetClassifier()
        model.load_state_dict(
            torch.load(
                "TextImagelatest.pth",
                map_location=torch.device("cpu"),
            )
        )

        model.eval()

        with torch.no_grad():
            output_text = model(
                image=img_tensor,
                text_input_ids=input_ids.unsqueeze(0),
                text_attention_mask=attention_mask.unsqueeze(0),
            )

        class_labels = [
            "TRUE",
            "SATIRE",
            "FALSE CONNECTION",
            "IMPOSTER CONTENT",
            "MANIPULATED CONTENT",
            "MISLEADING CONTENT",
        ]

        predicted_class = torch.argmax(output_text, dim=1).item()

        st.subheader("Predicted Category")
        st.success(class_labels[predicted_class])

    else:
        st.warning("Please upload an image.")