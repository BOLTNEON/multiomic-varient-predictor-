import torch
import torch.nn as nn

class MultiOmicsNN(nn.Module):
    def __init__(self, input_expr, input_prot):
        super(MultiOmicsNN, self).__init__()
        self.expr_layer = nn.Sequential(
            nn.Linear(input_expr, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.prot_layer = nn.Sequential(
            nn.Linear(input_prot, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, expr_input, prot_input):
        expr_out = self.expr_layer(expr_input)
        prot_out = self.prot_layer(prot_input)
        combined = torch.cat((expr_out, prot_out), dim=1)
        return self.classifier(combined)
