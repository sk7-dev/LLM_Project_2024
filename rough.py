import torch
import torch.nn as nn

class SimplifiedFFN(nn.Module):
    def __init__(self, n_embd=4):  # Using smaller embedding size for clarity
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),    # 4 -> 16
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),    # 16 -> 4
            nn.Dropout(0.2)
        )
        
        # Initialize weights for demonstration
        with torch.no_grad():
            # First layer weights (simplified for demonstration)
            self.net[0].weight = nn.Parameter(torch.tensor([
                [1.0, -0.5, 0.2, 0.1],   # First 4 neurons
                [-0.3, 0.8, -0.4, 0.6],
                [0.5, 0.2, -0.7, 0.3],
                [0.2, -0.3, 0.9, -0.5],
                [0.7, 0.4, -0.2, 0.8],   # Next 4 neurons
                [-0.4, 0.5, 0.3, -0.6],
                [0.3, -0.8, 0.5, 0.2],
                [0.6, 0.3, -0.4, 0.7],
                [-0.2, 0.6, 0.8, -0.3],  # Next 4 neurons
                [0.4, -0.2, -0.5, 0.9],
                [0.8, 0.7, 0.3, -0.4],
                [0.1, -0.6, 0.4, 0.5],
                [0.9, -0.4, 0.6, 0.2],   # Last 4 neurons
                [-0.5, 0.8, -0.3, 0.4],
                [0.2, 0.5, 0.7, -0.8],
                [0.6, -0.3, 0.2, 0.9]
            ]))
            
            # First layer bias
            self.net[0].bias = nn.Parameter(torch.tensor(
                [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4,
                 0.5, -0.5, 0.6, -0.6, 0.7, -0.7, 0.8, -0.8]
            ))
            
            # Second layer (projection back to embedding dimension)
            self.net[2].weight = nn.Parameter(torch.tensor([
                [0.2, -0.3, 0.1, -0.2, 0.4, -0.1, 0.3, -0.4, 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4, -0.2],
                [-0.1, 0.4, -0.2, 0.3, -0.3, 0.2, -0.4, 0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.4, 0.1],
                [0.3, -0.2, 0.4, -0.1, 0.2, -0.4, 0.1, -0.3, 0.4, -0.1, 0.2, -0.3, 0.1, -0.4, 0.3, -0.2],
                [-0.2, 0.1, -0.3, 0.4, -0.1, 0.3, -0.2, 0.4, -0.3, 0.2, -0.4, 0.1, -0.2, 0.3, -0.1, 0.4]
            ]))
            
            # Second layer bias
            self.net[2].bias = nn.Parameter(torch.tensor([0.1, -0.1, 0.2, -0.2]))

    def forward(self, x):
        print("\nInput:", x)
        
        # First linear layer
        linear1_out = self.net[0](x)
        print("\nAfter first linear layer:", linear1_out)
        
        # ReLU activation
        relu_out = torch.relu(linear1_out)
        print("\nAfter ReLU:", relu_out)
        
        # Second linear layer
        linear2_out = self.net[2](relu_out)
        print("\nAfter second linear layer:", linear2_out)
        
        # Dropout (only active during training)
        final_out = self.net[3](linear2_out)
        print("\nFinal output:", final_out)
        
        return final_out

# Example usage with different types of inputs
def demonstrate_ffn():
    ffn = SimplifiedFFN()
    ffn.eval()  # Set to evaluation mode to disable dropout
    
    print("Example 1: Standard positive input")
    x1 = torch.tensor([1.0, 2.0, 1.5, 0.5])
    _ = ffn(x1)
    
    print("\nExample 2: Mixed positive/negative input")
    x2 = torch.tensor([-1.0, 1.0, -0.5, 0.5])
    _ = ffn(x2)
    
    print("\nExample 3: Sequence of tokens")
    # Batch size = 1, sequence length = 2, embedding dim = 4
    x3 = torch.tensor([
        [[1.0, 1.0, 1.0, 1.0],  # First token
         [2.0, 2.0, 2.0, 2.0]]  # Second token
    ])
    _ = ffn(x3.reshape(-1, 4)).reshape(1, 2, 4)

demonstrate_ffn()