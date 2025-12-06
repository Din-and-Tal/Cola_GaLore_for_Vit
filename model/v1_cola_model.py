import math
import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers import ViTConfig, ViTPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput
from transformers.activations import ACT2FN
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTPooler

from .v1_cola_layer import ColaMDownProjLayer, ColaMUpProjLayer


class ColaViTConfig(ViTConfig):
    model_type = "cola_vit"

    def __init__(
        self,
        cola_rank_ratio=0.25,
        lr_act_type="gelu",  # Activation used inside the low-rank adapters
        only_lr_act=True,  # If True, relies primarily on the low-rank activation
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Default rank to 1/4th of hidden size if not specified
        self.rank = int(self.hidden_size * cola_rank_ratio)
        self.lr_act_type = lr_act_type
        self.only_lr_act = only_lr_act


class ColaViTPreAttn(nn.Module):
    """
    Computes the Down-Projections for Query, Key, and Value.
    Output: Low-rank representations of Q, K, V.
    """

    def __init__(self, config: ColaViTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Check if qkv bias is enabled in standard ViT (usually True)
        self.qkv_bias = config.qkv_bias

        # Down Projections (Input -> Low Rank)
        # Note: We use the activation specified in config for the bottleneck
        self.pre_q_proj = ColaMDownProjLayer(
            self.hidden_size,
            self.hidden_size,
            config.rank,
            lr_act_type=config.lr_act_type,
        )
        self.pre_k_proj = ColaMDownProjLayer(
            self.hidden_size,
            self.hidden_size,
            config.rank,
            lr_act_type=config.lr_act_type,
        )
        self.pre_v_proj = ColaMDownProjLayer(
            self.hidden_size,
            self.hidden_size,
            config.rank,
            lr_act_type=config.lr_act_type,
        )

    def forward(self, hidden_states):
        # Calculate low-rank states
        # Shape: [Batch, SeqLen, Rank]
        q_low = self.pre_q_proj(hidden_states)
        k_low = self.pre_k_proj(hidden_states)
        v_low = self.pre_v_proj(hidden_states)
        return q_low, k_low, v_low


class ColaViTSelfAttn(nn.Module):
    """
    Reconstructs Q, K, V (Up-Projection), performs Dot-Product Attention,
    and then Down-Projects the output.
    """

    def __init__(self, config: ColaViTConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.rank = config.rank

        # Up Projections (Low Rank -> Full Rank)
        self.post_q_proj = ColaMUpProjLayer(
            self.hidden_size, self.all_head_size, config.rank, bias=config.qkv_bias
        )
        self.post_k_proj = ColaMUpProjLayer(
            self.hidden_size, self.all_head_size, config.rank, bias=config.qkv_bias
        )
        self.post_v_proj = ColaMUpProjLayer(
            self.hidden_size, self.all_head_size, config.rank, bias=config.qkv_bias
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Output Down Projection (Full Rank -> Low Rank)
        # This replaces the first half of the 'Dense' output layer in standard ViT SelfOutput
        self.pre_output_proj = ColaMDownProjLayer(
            self.all_head_size,
            config.hidden_size,
            config.rank,
            lr_act_type=config.lr_act_type,
        )

    @property
    def hidden_size(self):
        return self.num_attention_heads * self.attention_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q_low, k_low, v_low, head_mask=None, output_attentions=False):
        # 1. Reconstruct Full Rank Q, K, V
        query_layer = self.post_q_proj(q_low)
        key_layer = self.post_k_proj(k_low)
        value_layer = self.post_v_proj(v_low)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # 2. Standard Attention Mechanism
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Flatten heads
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 3. Compress Output (Down Projection)
        output_low = self.pre_output_proj(context_layer)

        outputs = (output_low, attention_probs) if output_attentions else (output_low,)
        return outputs


class ColaViTPostAttn(nn.Module):
    """
    Reconstructs the Attention Output (Up-Projection) and handles the residual addition logic.
    """

    def __init__(self, config: ColaViTConfig):
        super().__init__()
        # Up Projection for the Output Dense layer
        self.post_output_proj = ColaMUpProjLayer(
            config.hidden_size, config.hidden_size, config.rank, bias=True
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, output_low):
        # Reconstruct full rank output
        hidden_states = self.post_output_proj(output_low)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ColaViTPreMLP(nn.Module):
    """
    Down-Projection for the first layer of the MLP (fc1).
    """

    def __init__(self, config: ColaViTConfig):
        super().__init__()
        self.pre_fc1 = ColaMDownProjLayer(
            config.hidden_size,
            config.intermediate_size,
            config.rank,
            lr_act_type=config.lr_act_type,
        )

    def forward(self, hidden_states):
        # Shape: [Batch, Seq, Rank]
        h_low_1 = self.pre_fc1(hidden_states)
        return h_low_1


class ColaViTSelfMLP(nn.Module):
    """
    Up-Projection for fc1 -> Activation -> Down-Projection for fc2.
    """

    def __init__(self, config: ColaViTConfig):
        super().__init__()
        self.config = config

        # Reconstruct intermediate state
        self.post_fc1 = ColaMUpProjLayer(
            config.hidden_size, config.intermediate_size, config.rank, bias=True
        )

        # Standard ViT activation (usually GELU)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        # Down project for fc2
        self.pre_fc2 = ColaMDownProjLayer(
            config.intermediate_size,
            config.hidden_size,
            config.rank,
            lr_act_type=config.lr_act_type,
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, h_low_1):
        # Reconstruct high dim intermediate
        hidden_states = self.post_fc1(h_low_1)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Compress again for fc2
        h_low_2 = self.pre_fc2(hidden_states)
        return h_low_2


class ColaViTPostMLP(nn.Module):
    """
    Up-Projection for fc2.
    """

    def __init__(self, config: ColaViTConfig):
        super().__init__()
        self.post_fc2 = ColaMUpProjLayer(
            config.intermediate_size, config.hidden_size, config.rank, bias=True
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, h_low_2):
        hidden_states = self.post_fc2(h_low_2)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ColaViTLayer(nn.Module):
    def __init__(self, config: ColaViTConfig):
        super().__init__()
        self.config = config

        # Attention parts
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.pre_attn = ColaViTPreAttn(config)
        self.self_attn = ColaViTSelfAttn(config)
        self.post_attn = ColaViTPostAttn(config)

        # MLP parts
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.pre_mlp = ColaViTPreMLP(config)
        self.self_mlp = ColaViTSelfMLP(config)
        self.post_mlp = ColaViTPostMLP(config)

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
    ):
        # --- Attention Block with CoLA-M Checkpointing ---

        # 1. Checkpoint 1: Prepare Low-Rank Q/K/V
        # We include LayerNorm in the checkpoint boundary
        def checkpoint_cola_attn_1(x):
            residual = x
            x = self.layernorm_before(x)
            q_low, k_low, v_low = self.pre_attn(x)
            return q_low, k_low, v_low, residual

        q_low, k_low, v_low, residual = torch.utils.checkpoint.checkpoint(
            checkpoint_cola_attn_1,
            hidden_states,
            preserve_rng_state=False,
            use_reentrant=True,
        )

        # 2. Checkpoint 2: Compute Attention (Self-Attn)
        # This recomputes Up-Projections inside the checkpoint during backward
        def checkpoint_cola_attn_2(q, k, v, hm):
            return self.self_attn(
                q, k, v, head_mask=hm, output_attentions=output_attentions
            )

        attn_outputs = torch.utils.checkpoint.checkpoint(
            checkpoint_cola_attn_2,
            q_low,
            k_low,
            v_low,
            head_mask,
            preserve_rng_state=False,
            use_reentrant=True,
        )

        output_low = attn_outputs[0]
        attn_weights = attn_outputs[1] if output_attentions else None

        # 3. Checkpoint 3: Post-Attn and Pre-MLP
        # Reconstruct Attn Output, Add Residual, Norm, Down-Project MLP
        def checkpoint_cola_attn_3_mlp_1(out_low, res):
            # Reconstruct and Add Residual (Attention Block End)
            attn_output = self.post_attn(out_low)
            hidden_states = res + attn_output

            # Start MLP Block (Norm + DownProj)
            residual_mlp = hidden_states
            hidden_states = self.layernorm_after(hidden_states)
            h_low_1 = self.pre_mlp(hidden_states)

            return h_low_1, residual_mlp

        h_low_1, residual_mlp = torch.utils.checkpoint.checkpoint(
            checkpoint_cola_attn_3_mlp_1,
            output_low,
            residual,
            preserve_rng_state=False,
            use_reentrant=True,
        )

        # --- MLP Block with CoLA-M Checkpointing ---

        # 4. Checkpoint 4: Self-MLP
        # UpProj FC1 -> Act -> DownProj FC2
        h_low_2 = torch.utils.checkpoint.checkpoint(
            self.self_mlp, h_low_1, preserve_rng_state=False, use_reentrant=True
        )

        # 5. Checkpoint 5: Post-MLP
        # UpProj FC2 -> Add Residual
        def checkpoint_cola_mlp_2(h_l2, res):
            mlp_output = self.post_mlp(h_l2)
            final_output = res + mlp_output
            return final_output

        layer_output = torch.utils.checkpoint.checkpoint(
            checkpoint_cola_mlp_2,
            h_low_2,
            residual_mlp,
            preserve_rng_state=False,
            use_reentrant=True,
        )

        outputs = (layer_output,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ColaViTEncoder(nn.Module):
    def __init__(self, config: ColaViTConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [ColaViTLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            # CoLA-M inherently uses checkpointing within the layer,
            # so standard gradient checkpointing flag might be redundant or conflicting
            # but we pass parameters normally.
            layer_outputs = layer_module(
                hidden_states,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ColaViTModel(ViTPreTrainedModel):
    config_class = ColaViTConfig

    def __init__(
        self, config: ColaViTConfig, add_pooling_layer=True, use_mask_token=False
    ):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config)
        self.encoder = ColaViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = None
        if add_pooling_layer:
            self.pooler = ViTPooler(config)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        # Pruning logic is complex with CoLA layers, leaving strict implementation out
        raise NotImplementedError("Head pruning not yet implemented for CoLA-ViT")

    def forward(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)

        if not return_dict:
            head_outputs = (
                (sequence_output, pooled_output)
                if pooled_output is not None
                else (sequence_output,)
            )
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ColaViTForImageClassification(ViTPreTrainedModel):
    config_class = ColaViTConfig

    def __init__(self, config: ColaViTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.cola_vit = ColaViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        self.post_init()

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.cola_vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # ViT classification usually uses the [CLS] token (index 0)
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
