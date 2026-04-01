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
    is replaced with the hidden state from the previous forward pass. This creates
    a continuous chain of thought that carries reasoning information.
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

        # Works with GPT-2 and can be extended to Llama etc.
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    @staticmethod
    def _to_legacy_cache(cache):
        """Convert KV cache to legacy tuple format: ((k, v), ...)"""
        if cache is None:
            return None
        # New DynamicCache format
        if hasattr(cache, 'key_cache'):
            return tuple(
                (k, v) for k, v in zip(cache.key_cache, cache.value_cache)
            )
        # Already legacy format (tuple of tuples, possibly with extra elements)
        return tuple((layer[0], layer[1]) for layer in cache)

    @staticmethod
    def _trim_cache(cache, trim_pos):
        """Trim KV cache to keep only tokens before trim_pos."""
        if cache is None:
            return None
        legacy = Coconut._to_legacy_cache(cache)
        return tuple(
            (k[:, :, :trim_pos, :], v[:, :, :trim_pos, :])
            for k, v in legacy
        )

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        """Forward pass with iterative latent hidden state feedback.

        Algorithm:
        1. Convert input_ids to embeddings
        2. Find all <|latent|> positions
        3. For each latent position:
           a. Forward pass up to that position (with KV cache for efficiency)
           b. Get last hidden state
           c. Replace the next latent token's embedding with that hidden state
        4. Final forward pass to get logits
        5. Compute cross-entropy loss (latent positions have label=-100, skipped)
        """
        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # [batch_size, list of latent positions per sample]

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache is None:
                # First forward pass: no KV cache yet
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
                # Subsequent passes: trim KV cache and reuse
                past_key_values = self._trim_cache(kv_cache, next_compute_range[0])

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # When using KV cache for the first k tokens,
                # outputs.hidden_states skips [0, k), so we need this offset
                # to correctly index the last hidden state

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
            kv_cache = self._to_legacy_cache(outputs.past_key_values)

            # Feedback: replace latent token embeddings with continuous thoughts
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # Break down into list of lists to avoid in-place operations
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # Replace latent embeddings with preceding hidden states
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # Reassemble
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # Final forward pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                self._trim_cache(kv_cache, next_compute_range[0])
                if kv_cache
                else None
            ),
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
        """Generate tokens after processing latent positions.

        1. Run forward() to fill latent positions with hidden state feedback
        2. Take argmax of last logit as first generated token
        3. Autoregressive loop for remaining tokens
        """
        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 for generation"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder, not used for loss
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
    """COCONUT with RL-based variable-length reasoning (future work).

    Will add a policy network that decides when to stop generating latent tokens.
    Currently a skeleton for future implementation.
    """

    def __init__(self, coconut_model, hidden_dim=768):
        super().__init__()
        self.coconut = coconut_model
        self.hidden_dim = hidden_dim

        # RL policy: given current hidden state, decide continue/stop
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # [continue_prob, stop_prob]
        )

    def forward(self, **kwargs):
        return self.coconut(**kwargs)

    def generate(self, **kwargs):
        return self.coconut.generate(**kwargs)
