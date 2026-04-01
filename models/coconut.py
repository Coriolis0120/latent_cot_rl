# -*- coding: utf-8 -*-
"""
COCONUT: Chain of Continuous Thought

Core mechanism:
1. Iteratively forward pass through the model
2. At each latent token position, take the last hidden state
3. Replace the next latent token's embedding with that hidden state
4. After all latent tokens are filled, generate the answer

Ported from official implementation:
https://github.com/facebookresearch/coconut
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class Coconut(nn.Module):
    """COCONUT model wrapping a base causal LM (GPT-2).

    The key innovation is the iterative forward pass with hidden state feedback:
    instead of treating <|latent|> as regular tokens, each latent token's embedding
    is replaced with the hidden state from the previous forward pass.
    """

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):
        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    @staticmethod
    def _trim_cache(cache, trim_pos):
        """Trim KV cache to keep only tokens before trim_pos.

        Maintains the original cache format (DynamicCache stays DynamicCache,
        legacy tuple stays tuple) so it remains compatible with the model.
        """
        if cache is None:
            return None
        # DynamicCache / new format: trim in-place
        if hasattr(cache, "key_cache"):
            for i in range(len(cache.key_cache)):
                cache.key_cache[i] = cache.key_cache[i][:, :, :trim_pos, :]
                cache.value_cache[i] = cache.value_cache[i][:, :, :trim_pos, :]
            if hasattr(cache, "_seen_tokens"):
                cache._seen_tokens = trim_pos
            return cache
        # Legacy tuple format: each layer entry is a tuple of N tensors
        # (handles (k, v) and (k, v, extra...) formats)
        trimmed = []
        for layer_entry in cache:
            if isinstance(layer_entry, (tuple, list)):
                trimmed.append(
                    tuple(t[:, :, :trim_pos, :] for t in layer_entry)
                )
            else:
                trimmed.append(layer_entry)
        return tuple(trimmed)

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        """Forward pass with iterative latent hidden state feedback."""
        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                self._trim_cache(kv_cache, next_compute_range[0])

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # Feedback: replace latent token embeddings with continuous thoughts
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # Final forward pass
        if kv_cache is not None:
            self._trim_cache(kv_cache, next_compute_range[0])

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=kv_cache,
            output_hidden_states=True,
        )

        logits.append(outputs.logits)
        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # not used internally, kept for API compatibility
        max_new_tokens=16,
        output_embedding=False,
        **kwargs
    ):
        """Generate tokens after processing latent positions."""
        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 for generation"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # Get first generated token
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # Autoregressive generation
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if output_embedding:
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)


class CoconutWithRL(nn.Module):
    """COCONUT with RL-based variable-length reasoning (future work)."""

    def __init__(self, coconut_model, hidden_dim=768):
        super().__init__()
        self.coconut = coconut_model
        self.hidden_dim = hidden_dim

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, **kwargs):
        return self.coconut(**kwargs)

    def generate(self, **kwargs):
        return self.coconut.generate(**kwargs)
