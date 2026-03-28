import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
from cache_utils import Cache
from transformers.activations import ACT2FN

class LlamaDecoderLayerWrapper(nn.Module):
    def __init__(self, llama_decoder_layer, tsv_layer, model_name='llama3.1-8B'):
        super().__init__()
        self.llama_decoder_layer = llama_decoder_layer
        self.tsv_layer = tsv_layer  # Instance of ICVLayer
        self.model_name = model_name

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    )-> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Save original residual state
        residual = hidden_states

        # Forward pass through the input layer norm
        hidden_states = self.llama_decoder_layer.input_layernorm(hidden_states)


        if self.model_name == 'qwen2.5-7B':
            hidden_states, self_attn_weights, present_key_value = self.llama_decoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            
        else:    
            hidden_states, self_attn_weights, present_key_value = self.llama_decoder_layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
        
        # Add residual + steering vector after self-attention
        hidden_states = residual.to(hidden_states.device) + hidden_states
        

        # Save residual state for the MLP
        residual = hidden_states

        # Forward pass through the post-attention layer norm and MLP
        hidden_states = self.llama_decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.llama_decoder_layer.mlp(hidden_states)

        # Add residual + steering vector after MLP
        hidden_states = residual + hidden_states
        hidden_states = self.tsv_layer(hidden_states)  # Add steering vector

        # Return the outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs
        
class TSVLayer(nn.Module):

    def __init__(self, tsv, lam):
        super(TSVLayer, self).__init__()
        self.tsv = tsv
        self.lam = lam

    def forward(self, x):
        if self.tsv is not None:

            x = x.half()
            y = self.lam[0] * self.tsv.repeat(1,x.shape[1],1)
            y = y.to(x.device)
            x = x.half() + y
            
            return x.half()
        
        else:
            
            return x.half()
        

def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)


def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def get_embedding_layer(model: PreTrainedModel):

    keywords = ["emb", "wte"]
    return find_module(model, keywords)


def get_lm_head(model: PreTrainedModel):
    keywords = ["lm_head", "embed_out"]
    return find_module(model, keywords)


def get_lm_pipeline(model: PreTrainedModel):
    model_class = model.__class__.__name__

    if model_class == "LlamaForCausalLM":
        return nn.Sequential(model.model.norm, model.lm_head)
    elif model_class == "RWForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoXForCausalLM":
        return nn.Sequential(model.gpt_neox.final_layer_norm, model.embed_out)

    # TODO: make the default case more robust
    return get_lm_head(model)


def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path


def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def add_tsv_layers(model: PreTrainedModel, tsv: Tensor, alpha: list, args):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    attn_keywords = ["self_attn"]
    
    assert len(tsv) == len(layers)
    if args.component == 'mlp':
        for i, layer in enumerate(layers):
            if i == args.str_layer:
                original_mlp = find_module(layer, mlp_keywords)
                layer.mlp = nn.Sequential(original_mlp, TSVLayer(tsv[i], alpha)) 

    elif args.component == 'attn':
        for i, layer in enumerate(layers):
            if i == args.str_layer:
                original_attn = find_module(layer, attn_keywords)
                layer.self_attn = nn.Sequential(original_attn, TSVLayer(tsv[i], alpha)) 
                
    elif args.component == 'res':
        
        for i, layer in enumerate(layers):
            if i == args.str_layer:
                decoder_layer = layers[i]
                layers[i] = LlamaDecoderLayerWrapper(decoder_layer, TSVLayer(tsv[i], alpha), args.model_name)