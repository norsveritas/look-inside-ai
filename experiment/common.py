"""Shared helpers: loading GPT-2 and recording its internal activations."""
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CHARTS = os.path.join(ROOT, "charts")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CHARTS, exist_ok=True)

N_LAYERS = 12

# Chart colours (validated default palette, light mode)
BLUE, ORANGE, AQUA, GREY = "#257de0", "#f47230", "#00b894", "#9a9a9a"
INK, INK2 = "#191919", "#666666"


def load_model():
    """Load GPT-2 small (124M parameters) with TransformerLens, on the CPU."""
    from transformer_lens import TransformerBridge

    model = TransformerBridge.boot_transformers("gpt2", device="cpu")
    model.enable_compatibility_mode()  # gives the classic HookedTransformer hook names
    model.eval()
    return model


def record(model, sentences):
    """Run each sentence through the model and keep two kinds of activations.

    resid[i, layer, :]  = residual stream (the model's running "notes") entering each layer,
                          plus the final output (index 12), averaged over the sentence's words.
    neurons[i, layer, n] = strongest firing of each MLP neuron anywhere in the sentence.
    """
    names = [f"blocks.{l}.hook_resid_pre" for l in range(N_LAYERS)] + [
        f"blocks.{N_LAYERS - 1}.hook_resid_post"
    ]
    neuron_names = [f"blocks.{l}.mlp.hook_post" for l in range(N_LAYERS)]
    resid, neurons, token_acts = [], [], []
    for s in sentences:
        toks = model.to_tokens(s)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names or n in neuron_names)
        # skip position 0 (the start-of-text token, which is identical for every sentence)
        resid.append(torch.stack([cache[n][0, 1:].mean(0) for n in names]).numpy())
        nact = torch.stack([cache[n][0, 1:] for n in neuron_names])  # layer, pos, neuron
        neurons.append(nact.max(1).values.numpy())
        token_acts.append((model.to_str_tokens(s)[1:], nact.numpy()))
    return np.array(resid), np.array(neurons), token_acts


def kurdish_token_ids(model):
    words = [" Kurdish", " Kurds", " Kurd", " Kurdistan"]
    return [model.to_single_token(w) for w in words], words
