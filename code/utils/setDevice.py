import torch

def get_device():
    # NVIDIA GPU 사용
    if torch.cuda.is_available():
        return torch.device("cuda")  
    
     # Apple M1/M2 GPU 사용
    elif torch.backends.mps.is_available():
        return torch.device("mps") 
    
    # CPU 사용
    else:
        return torch.device("cpu")  

device = get_device()

if __name__=="__main__":
    device = get_device()
    print(f"Using device: {device}")