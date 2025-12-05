import re
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_weights_from_csharp(csharp_text):
    """Parse weight data from C# code"""
    #  Remove annotation
    csharp_text = re.sub(r'/\*.*?\*/', '', csharp_text, flags=re.DOTALL)
    csharp_text = re.sub(r'//.*', '', csharp_text)
    
    def extract(name, n):
        # regular expression
        patterns = [
            rf'private\s+static\s+float\[\]\s+{name}\s*=\s*\{{([^}}]+)\}}',
            rf'float\[\]\s+{name}\s*=\s*\{{([^}}]+)\}}',
            rf'{name}\s*=\s*\{{([^}}]+)\}}',
            rf'{name}\s*=\s*new\s+float\[\]\s*\{{([^}}]+)\}}'
        ]
        
        for pat in patterns:
            m = re.search(pat, csharp_text, re.DOTALL | re.IGNORECASE)
            if m:
                data = m.group(1).replace('f', ' ').replace('F', ' ')
                vals = [float(x) for x in re.split(r'[,\s]+', data) if x.strip()]
                if len(vals) == n:
                    logger.info(f"Found {name} with {len(vals)} values")
                    return vals
                else:
                    logger.warning(f"Found {name} but length {len(vals)} != expected {n}")
        
        # if can not fine,print this ↓
        logger.error(f"Could not find {name} in C# code")
        logger.error("Available arrays found:")
        for pattern in [r'(\w+)\s*=\s*\{', r'float\[\]\s+(\w+)\s*=']:
            matches = re.findall(pattern, csharp_text)
            for match in matches:
                logger.error(f"  - {match}")
        
        raise AssertionError(f"{name} not found")
    
    return (
        extract('weights1', 320), extract('bias1', 32),
        extract('weights2', 512), extract('bias2', 16),
        extract('weights3', 48),  extract('bias3', 3)
    )

def load_existing_weights(model, weights1, bias1, weights2, bias2, weights3, bias3, device=None):
    """Load weights into model - Fixed device handling"""
    if device is None:
        device = next(model.parameters()).device
    
    with torch.no_grad():
        model.layer1.weight.data = torch.tensor(weights1, device=device).view(10, 32).T
        model.layer1.bias.data   = torch.tensor(bias1, device=device)
        model.layer2.weight.data = torch.tensor(weights2, device=device).view(32, 16).T
        model.layer2.bias.data   = torch.tensor(bias2, device=device)
        model.layer3.weight.data = torch.tensor(weights3, device=device).view(16, 3).T
        model.layer3.bias.data   = torch.tensor(bias3, device=device)

def auto_mix(output_rgb, bg_colour, acc_a):
    """
    

    Automatic foreground-background blending to generate final RGBA
    output_rgb: [batch, 3], model output
    bg_colour: [batch, 3] or [3], background color
    acc_a: [batch, 1] or [1], blending weight
    Returns: [batch, 4], blended RGBA
    """
    device = output_rgb.device
    
    if bg_colour.dim() == 1:
        bg_colour = bg_colour.unsqueeze(0).expand_as(output_rgb)
    if acc_a.dim() == 1:
        acc_a = acc_a.unsqueeze(0).expand(output_rgb.shape[0], 1)
    
    # Ensure all tensors are on the same device
    bg_colour = bg_colour.to(device)
    acc_a = acc_a.to(device)
    
    final_rgb = output_rgb + acc_a * bg_colour
    alpha = acc_a  
    output_rgba = torch.cat([final_rgb, alpha], dim=1)
    return output_rgba

def export_weights_to_csharp(model, output_path='ConsistentRGBAWeights.cs'):
    """Export consistent weights to C# format"""
    with torch.no_grad():
        w1 = model.layer1.weight.data.T.flatten().cpu().numpy()
        b1 = model.layer1.bias.data.cpu().numpy()
        w2 = model.layer2.weight.data.T.flatten().cpu().numpy()
        b2 = model.layer2.bias.data.cpu().numpy()
        w3 = model.layer3.weight.data.T.flatten().cpu().numpy()
        b3 = model.layer3.bias.data.cpu().numpy()
    
    csharp_code = f'''using UnityEngine;

namespace DFAOIT
{{
    /// <summary>
    /// Consistent RGBA prediction weights
    /// Ensures input-output consistency for RGBA prediction
    /// </summary>
    public static class ConsistentRGBAWeights
    {{
        // Layer 1: 10 -> 32
        private static float[] weights1 = {{ {', '.join([f'{w:.6f}f' for w in w1])} }};
        private static float[] bias1 = {{ {', '.join([f'{b:.6f}f' for b in b1])} }};
        
        // Layer 2: 32 -> 16
        private static float[] weights2 = {{ {', '.join([f'{w:.6f}f' for w in w2])} }};
        private static float[] bias2 = {{ {', '.join([f'{b:.6f}f' for b in b2])} }};
        
        // Layer 3: 16 -> 3
        private static float[] weights3 = {{ {', '.join([f'{w:.6f}f' for w in w3])} }};
        private static float[] bias3 = {{ {', '.join([f'{b:.6f}f' for b in b3])} }};
        
        /// <summary>
        /// Apply consistent weights to material
        /// </summary>
        public static void ApplyToMaterial(Material material)
        {{
            material.SetFloatArray("_Weights1", weights1);
            material.SetFloatArray("_Bias1", bias1);
            material.SetFloatArray("_Weights2", weights2);
            material.SetFloatArray("_Bias2", bias2);
            material.SetFloatArray("_Weights3", weights3);
            material.SetFloatArray("_Bias3", bias3);
        }}
    }}
}}'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(csharp_code)
    
    logger.info(f"Consistent RGBA weights exported to {output_path}")