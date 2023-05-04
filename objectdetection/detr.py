import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import  math
class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=128, nheads=8, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
        super(DETR, self).__init__()
        self.hidden_dim = hidden_dim

        # backbone CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim, dropout)
        self.proj = nn.Linear(512, hidden_dim)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # object detection heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # initialize parameters
        self.init_weights()

    def forward(self, inputs):
        # backbone CNN
        features = self.backbone(inputs)

        # flatten and permute features
        bs, c, h, w = features.shape
        features = features.view(bs, c, -1).permute(0, 2, 1)

        # positional encoding
        features = self.proj(features)
        features = self.pos_enc(features)

        # transformer encoder
        memory = self.transformer_encoder(features).to(inputs.device)

        # transformer decoder
        query = torch.zeros(bs, h * w, self.hidden_dim).to(inputs.device)  # query size based on the image size
        tgt = {'class': torch.zeros(bs, 100), 'bbox': torch.zeros(bs, 100, 4)}
        output = self.transformer_decoder(query, memory, memory_key_padding_mask=None,
                                          memory_mask=None)

        # object detection heads
        classes = self.class_embed(output)
        bboxes = self.bbox_embed(output)

        # output dictionary
        out = {'pred_logits': classes[-1], 'pred_boxes': bboxes[-1]}
        return out

    def generate_square_subsequent_mask(self, sz):
        """Generate mask for Transformer decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # register buffer
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

