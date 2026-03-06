# -*- coding: utf-8 -*-
"""
Model Configuration Mapping Table
Handles attribute name differences across model architectures
Only add configurations that differ from default
Keys are matched by model_id (folder name) or model_type (from config.json)

Note: Many model config classes automatically map legacy attribute names to standard names.
For example, BloomConfig maps 'n_embed' to 'hidden_size', so no special mapping is needed.
"""

# Layer count configuration key (from config)
MODEL_LAYER_KEY = {
    'default': 'num_hidden_layers',
    # GPT-2 family (model_type: gpt2) - uses n_layer
    'gpt2': 'n_layer',
}

# Hidden size configuration key (from config)
MODEL_HIDEEN_SIZE_KEY = {
    'default': 'hidden_size',
    # GPT-2 family (model_type: gpt2) - uses n_embd
    'gpt2': 'n_embd',
}

# Data type configuration key (from config)
MODEL_DTYPE_KEY = {
    'default': 'dtype',
}

# Embedding layer configuration key (from model structure)
MODEL_EMBED_KEY = {
    'default': 'embed_tokens',
    # BERT family (model_type: bert)
    'bert': 'embeddings',
    'roberta': 'embeddings',
    'distilbert': 'embeddings',
    # GPT-2 family (model_type: gpt2)
    'gpt2': 'wte',
    # T5 family (model_type: t5)
    't5': 'shared',
}

# Rotary embedding configuration key (from model structure)
MODEL_ROTATE_KEY = {
    'default': 'rotary_emb',
    # BERT family (no rotary, model_type: bert)
    'bert': 'embeddings',
    'roberta': 'embeddings',
    'distilbert': 'embeddings',
    # GPT-2 family (no rotary, model_type: gpt2)
    'gpt2': 'wpe',
    # T5 family (no rotary, model_type: t5)
    't5': 'shared',
}

# Transformer Block configuration key (from model structure)
MODEL_TRANS_KEY = {
    'default': 'layers',
    # BERT family (model_type: bert)
    'bert': 'encoder.layer',
    'roberta': 'encoder.layer',
    'distilbert': 'transformer.layer',
    # GPT-2 family (model_type: gpt2)
    'gpt2': 'h',
    # T5 family (model_type: t5)
    't5': 'encoder.block',
}
