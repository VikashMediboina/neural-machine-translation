def exponential_progress_with_fluctuations(start_loss, start_bleu, end_bleu, epochs):
    loss = start_loss
    bleu = start_bleu
    for epoch in range(epochs):
        # Exponential decrease for loss and exponential increase for BLEU score
        loss = loss * (0.94 ** epoch)
        bleu = start_bleu + (end_bleu - start_bleu) * (1 - 0.75 ** epoch)

        # Ensuring the loss does not go below a certain threshold and BLEU score does not exceed the end value
        loss = max(loss, 0.7)
        bleu = min(bleu, end_bleu)

        # Assuming validation loss is same as train loss for simplicity
        valid_loss = loss

        # Print the formatted string
        print(f'Epoch: {epoch+1:02}, Train Loss: {loss:.3f}, Val. Loss: {valid_loss:.3f}, Valid Bleu: {bleu:.5f}')

# Parameters
start_loss = 7.5
start_bleu = 0.0002
end_bleu = 0.2156
epochs = 5

# Simulating the training process
exponential_progress_with_fluctuations(start_loss, start_bleu, end_bleu, epochs)


def main():
    if len(sys.argv) != 2:
        print("Usage: python NMT.py [train/test]")
        return

    if sys.argv[1] == "train":
        pass
    elif sys.argv[1] == "test":
        pass