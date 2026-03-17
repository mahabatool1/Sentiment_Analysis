The Amazon Review Sentiment Dashboard is an interactive web application built with Gradio and hosted on Hugging Face Spaces. It leverages a fine‑tuned transformer model to analyze customer reviews and classify them into Negative, Neutral, or Positive sentiments. The dashboard provides real‑time predictions with probability scores, making it easy to understand customer feedback at a glance.
Key Features
- 📝 Customer Review Input: Users can paste any review into the text box.
- 📊 Prediction Probabilities: Displays confidence scores for each sentiment class.
- 🎨 Visual Output: Highlights the final sentiment with color coding (red for negative, yellow for neutral, green for positive).
- 🌐 Permanent Hosting: Deployed on Hugging Face Spaces, accessible anytime without requiring Colab or Kaggle runtime.
- ⚡ Fast & Lightweight: Uses the cardiffnlp/twitter-roberta-base-sentiment model for efficient inference.
Use Cases
- 📦 E‑commerce: Quickly assess customer satisfaction from product reviews.
- 📈 Business Analytics: Identify trends in feedback to improve services.
- 🎓 Learning Demo: Showcases practical application of NLP and Gradio deployment.
Tech Stack
- Python
- Transformers (Hugging Face)
- Torch
- Gradio
- Hugging Face Spaces (deployment)
