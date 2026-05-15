import torch
from transformers import RobertaTokenizer

from config import CFG
from models import FrozenRobertaBiGRU


LABEL_MAP = {
    0: "Factual",
    1: "Hallucination"
}


def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(CFG.model_name)

    model = FrozenRobertaBiGRU(
        model_name=CFG.model_name,
        num_classes=CFG.num_classes,
        hidden_dim=CFG.rnn_hidden,
        num_layers=CFG.rnn_layers,
        dropout=CFG.rnn_dropout
    )

    model.load_state_dict(
        torch.load(CFG.best_model_path, map_location=CFG.device)
    )

    model.to(CFG.device)
    model.eval()

    return model, tokenizer


def predict(text):
    model, tokenizer = load_model()

    encoding = tokenizer(
        text,
        max_length=CFG.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(CFG.device)
    attention_mask = encoding["attention_mask"].to(CFG.device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return {
        "prediction": LABEL_MAP[predicted_class],
        "confidence": probabilities[0][predicted_class].item()
    }


if __name__ == "__main__":
    sample_text = """
    Passage: The Eiffel Tower is located in Paris.
    Question: Where is the Eiffel Tower located?
    Answer: The Eiffel Tower is located in London.
    """

    result = predict(sample_text)
    print(result)
