import torch

class CFG:
    model_name = "roberta-base"
    max_len = 256
    batch_size = 32
    seed = 42

    epochs = 5
    learning_rate = 2e-4

    rnn_hidden = 128
    rnn_layers = 2
    rnn_dropout = 0.3

    num_classes = 2
    save_dir = "saved_models"
    best_model_path = "saved_models/lightweight_hallucination_detector.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
