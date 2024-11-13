import numpy as np
import torch

def vist_voice(text_vits, tokenizer, model):
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tokenize the input text and move tensors to the device
    inputs = tokenizer(text_vits, return_tensors="pt").to(device)
    
    # Generate waveform from VitsModel
    with torch.no_grad():
        output_vits = model(**inputs).waveform
        output_vits = output_vits.squeeze().cpu().numpy()  # Move the output to CPU before converting to numpy
    
    return output_vits
