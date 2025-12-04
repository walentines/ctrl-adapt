import torch
import torch.nn as nn

class ControlNetFusion(nn.Module):
    def __init__(self, channels_per_layer, fusion_method="conv"):
        """
        :param channels_per_layer: List of output channel sizes at each layer.
        :param fusion_method: "conv" (Concat + Conv), "cross_attn" (Transformer), or "adaptive" (Weighted Sum).
        """
        super(ControlNetFusion, self).__init__()
        self.fusion_method = fusion_method

        if fusion_method == "conv":
            self.fusion_layers = nn.ModuleList([
                nn.Conv2d(in_channels=c * 2, out_channels=c, kernel_size=1)  # Reduce back to original shape
                for c in channels_per_layer
            ])
        
        elif fusion_method == "cross_attn":
            self.attn_layers = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=c, num_heads=8, batch_first=True)
                for c in channels_per_layer
            ])
            self.final_convs = nn.ModuleList([
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1)  # Keep channel size
                for c in channels_per_layer
            ])
        
        elif fusion_method == "adaptive":
            self.weight_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c * 2, c, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(c, 1, kernel_size=1),
                    nn.Sigmoid()
                ) for c in channels_per_layer
            ])
        
        self.dtype=torch.float32
        

    def forward(self, seg_features, depth_features):
        """
        :param seg_features: List of feature maps from segmentation ControlNet.
        :param depth_features: List of feature maps from depth ControlNet.
        :return: List of fused feature maps, each with the same shape as inputs.
        """
        fused_features = []

        for i in range(len(seg_features)):
            f_seg, f_depth = seg_features[i], depth_features[i]  # Same shape

            if self.fusion_method == "conv":
                x = torch.cat([f_seg, f_depth], dim=1)  # Concatenation
                fused = self.fusion_layers[i](x)  # 1x1 Conv to reduce back to original channels
            
            elif self.fusion_method == "cross_attn":
                B, C, H, W = f_seg.shape
                f_seg_flat = f_seg.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
                f_depth_flat = f_depth.view(B, C, -1).permute(0, 2, 1)
                fused, _ = self.attn_layers[i](f_seg_flat, f_depth_flat, f_depth_flat)
                fused = fused.permute(0, 2, 1).view(B, C, H, W)
                fused = self.final_convs[i](fused)  # Ensure same channel size
            
            elif self.fusion_method == "adaptive":
                x = torch.cat([f_seg, f_depth], dim=1)
                weight = self.weight_layers[i](x)  # Learnable weight
                # fused = weight * f_seg + (1 - weight) * f_depth  # Weighted sum
                fused = f_depth # pui f_depth pt depth doar

            fused_features.append(fused)

        return fused_features

# # Example Usage
# shapes = [
#     (320, 64, 96), (320, 64, 96), (320, 64, 96),
#     (320, 32, 48), (640, 32, 48), (640, 32, 48),
#     (640, 16, 24), (1280, 16, 24), (1280, 16, 24),
#     (1280, 8, 12), (1280, 8, 12), (1280, 8, 12)
# ]

# seg_features = [torch.randn(1, c, h, w) for c, h, w in shapes]
# depth_features = [torch.randn_like(f) for f in seg_features]  # Same shapes

# fusion_model = ControlNetFusion(
#     channels_per_layer=[s[0] for s in shapes],  # Extract just channel sizes
#     fusion_method="adaptive"  # Choose "conv", "cross_attn", or "adaptive"
# )

# fused_features = fusion_model(seg_features, depth_features)

# # Check if output matches expected shapes
# for i, (f, s) in enumerate(zip(fused_features, shapes)):
#     print(f"Fused Layer {i} Shape: {f.shape} (Expected: {s})")