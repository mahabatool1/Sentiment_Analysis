import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax

# Load model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

labels = ['Negative', 'Neutral', 'Positive']
colors = {"Negative":"#FF9999", "Neutral":"#FFFF99", "Positive":"#99FF99"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = outputs.logits[0].detach().numpy()
    probs = softmax(scores)
    pred = labels[torch.argmax(outputs.logits).item()]
    
    result = {labels[i]: float(probs[i]) for i in range(len(labels))}
    styled_pred = f"<div style='text-align:center; font-size:22px; font-weight:bold; color:{colors[pred]};'>{pred}</div>"
    return result, styled_pred

# Build polished Gradio dashboard
with gr.Blocks(theme="soft") as demo:
    with gr.Column(elem_id="main-col"):
        gr.Markdown("<h1 style='text-align:center; color:#4A4A4A;'>🌸 Amazon Review Sentiment Dashboard</h1>")
        gr.Markdown("<p style='text-align:center;'>Paste any customer review and see predicted sentiment in real time.</p>")
        
        with gr.Row():
            review_input = gr.Textbox(lines=3, placeholder="Enter a review...", label="Customer Review")
        
        with gr.Row():
            sentiment_probs = gr.Label(num_top_classes=3, label="Prediction Probabilities")
        
        with gr.Row():
            sentiment_output = gr.HTML(label="Final Sentiment")
        
        with gr.Row():
            submit_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
        
        submit_btn.click(predict_sentiment, inputs=review_input, outputs=[sentiment_probs, sentiment_output])

demo.launch()