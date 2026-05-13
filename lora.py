import torch

class LoRALayer(torch.nn.Module):
    def __init__(self, original_layer, alpha, rank, dropout=0.0):
        super(LoRALayer, self).__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        
        self.lora_A = torch.nn.Parameter(torch.randn(original_layer.in_features, rank) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, original_layer.out_features))

        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        base_out = self.original_layer(x)
        lora_out = (self.dropout(x).float() @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)
    

def inject_lora_layers(model, rank, alpha, dropout, target_modules=["q_proj", "v_proj"]):
    for name, module in list(model.named_modules()):
        if any(target_module in name for target_module in target_modules):
            if isinstance(module, LoRALayer):
                continue

            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)

            lora_layer = LoRALayer(module, rank, alpha, dropout)
            setattr(parent, child_name, lora_layer)
    
    return model