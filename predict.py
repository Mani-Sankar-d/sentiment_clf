import torch
from transformers import BertTokenizer

from model.bert import Model
from utils.checkpoint import load_ckpt

import pandas as pd
 
df = pd.read_csv("sentiment_data/train.csv")
# print(df[df["sentiment"] == "negative"].shape)

@torch.no_grad()
def predict_sentiment(model, tokenizer, device, text, max_len=512):
    """
    Predict sentiment for a single text input.
    
    Args:
        model: Trained sentiment model
        tokenizer: BERT tokenizer
        device: torch device
        text: Input text string
        max_len: Maximum sequence length
    
    Returns:
        prediction: 0 (Negative) or 1 (Positive)
        confidence: Softmax probability of predicted class
    """
    model.eval()
    
    # Tokenize the input text
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Get model prediction
    outputs = model(input_ids, attention_mask)
    probs = torch.softmax(outputs, dim=-1)
    pred = torch.argmax(outputs, dim=-1).item()
    confidence = probs[0, pred].item()
    
    return pred, confidence

def main():
    # ---- Setup ----
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    model = Model(vocab_size=tokenizer.vocab_size, embed_dim=256, max_len=512, n_heads=8, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters())  # dummy optimizer for checkpoint loading
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ---- Load best checkpoint ----
    print("Loading model checkpoint...")
    model, optimizer, _, _, _ = load_ckpt(model=model, optimizer=optimizer, path="best_checkpoint.pth", device=device)
    print(f"Model loaded successfully on {device}\n")

    # ---- Interactive loop ----
    print("=" * 60)
    print("📝 Sentiment Analysis - Interactive Mode")
    print("=" * 60)
    print("Type your text and press Enter to get sentiment prediction.")
    print("Type 'quit' or 'exit' to stop.\n")
    tp=0
    fp=0
    tn=0
    fn=0

    for i in range(len(df)):
        # Get user input
        user_input = df.loc[i,"review"]
        truth = df.loc[i,"sentiment"]
        # Check for exit command
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        # Skip empty input
        if not user_input:
            print("⚠️ Please enter some text.\n")
            continue
        
        # Make prediction
        pred, confidence = predict_sentiment(model, tokenizer, device, user_input)
        
        # Display result
        label = "positive" if pred == 1 else "negative"
        if(label == "positive" and truth=="positive"): tp+=1
        elif(label == "positive" and truth=="negative"): fp+=1
        elif(label == "negative" and truth=="positive"): fn+=1
        elif(label == "negative" and truth=="negative"): tn+=1

        # print(f"\n{'─' * 60}")
        # print(f"📌 Input: {user_input}")
        # print(f"➡️ Prediction: {label}")
        # print(f"📊 Confidence: {confidence:.2%}")
        # print(f"{'─' * 60}\n")
    if (tp+fp) > 0 and (tp+fn) > 0:
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = (2*p*r)/(p+r)
        print(f"Precision: {p:.4f}")
        print(f"Recall: {r:.4f}")
        print(f"F1-score: {f1:.4f}")
    else:
        print("Cannot calculate F1-score (division by zero)")

if __name__ == "__main__":
    main()