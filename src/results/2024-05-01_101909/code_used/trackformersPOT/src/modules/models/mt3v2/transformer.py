import torch 
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from util.misc import inverse_sigmoid
#from modules.transformer import PreProccessor, TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder 
import copy
import math
from typing import Optional, List
import numpy as np


class PreProccessor(nn.Module):
    def __init__(self, d_model, d_detections, normalization_constant, use_fourier_feat=False, gauss_scale=1.0):
        super().__init__()
        self.normalization_constant = normalization_constant
        self.use_fourier_feat = use_fourier_feat
        if use_fourier_feat:
            B = torch.empty((d_detections, d_model // 2)).normal_()
            B = B * gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_detections = d_detections
            self.d_model = d_model
            self.use_fourier_feat = use_fourier_feat
            self.linear1 = nn.Linear(d_model, d_model, bias=False)
        else:
            self.linear1 = nn.Linear(d_detections, d_model, bias=False)

    def forward(self, src):
        out = src / self.normalization_constant
        if self.use_fourier_feat:
            bs, num_batch_max_meas, d_detections = src.shape
            d_in = self.gauss_B.shape[0]
            d_out = self.d_model // 2
            out = src * 2 * np.pi
            out = torch.mm(out.view(-1, d_in), self.gauss_B[:, :d_out]).view(
                bs, num_batch_max_meas, d_out
            )
            final_embeds = [out.sin(), out.cos()]
            out = torch.cat(final_embeds, dim=2)
        return self.linear1(out)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, false_detect_embedding=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.false_detect_embedding = false_detect_embedding

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
        

    def forward(self, src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None):
    # Normalize input before attention operations if normalize_before is True
        if self.normalize_before:
            src2 = self.norm1(src)  # Pre-normalization
            # Positional embedding is added to the input sequence
            q = k = self.with_pos_embed(src2, pos)
        else:
            q = k = self.with_pos_embed(src, pos)

        # If false detection embedding is enabled, process it accordingly
        if self.false_detect_embedding:
            n, bs, dim = q.shape  # Original dimensions of query/key
            nf, _ = self.false_detect_embedding.weight.shape  # False detect embedding dimensions
            # Repeat false detection embedding for batch size and permute for correct shape
            false_detect_embedding = self.false_detect_embedding.weight.unsqueeze(-1).permute(0, 2, 1).repeat(1, bs, 1)
            # Concatenate the false detection embedding to query and key (and value if pre-normalization is applied)
            k = torch.cat((k, false_detect_embedding), dim=0)
            val = torch.cat((src, false_detect_embedding), dim=0) if self.normalize_before else src
            # Create a mask for false detection tokens to ignore them in attention computations
            false_detect_mask = torch.zeros(bs, nf).bool().to(src_key_padding_mask.device)
            src_key_padding_mask = torch.cat((src_key_padding_mask, false_detect_mask), dim=1)
        else:
            val = src  # Use original source as value if false detection embedding is not used
        
        # Self-attention operation
        src2 = self.self_attn(q, k, value=val, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]
        # Add and norm step after self-attention
        src = src + self.dropout1(src2)
        
        
        if self.normalize_before:
            # Post-normalization if normalize_before is True
            src2 = self.norm2(src)
            # Feed-forward network operations
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))    
        else:
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))    
        
        # Add and norm step for feed-forward network
        src = src + self.dropout2(src2)
        
        if not self.normalize_before: 
            src = self.norm2(src)
        
        return src

    


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, false_detect_embedding=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        if false_detect_embedding:
            for layer in self.layers:
                layer.false_detect_embedding = false_detect_embedding
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
            # Normalize the target before processing, if specified.
            if self.normalize_before:
                tgt2 = self.norm1(tgt)  # Apply first layer normalization if normalization is done before attention.
            else:
                tgt2 = tgt  # Use the target tensor as is if normalization is not done before attention.

            # Apply positional embeddings to target for self-attention.
            q = k = self.with_pos_embed(tgt2, query_pos)  # Embed positions into queries and keys.

            # Perform self-attention on the target.
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]  # The first element is the attention output.
            tgt = tgt + self.dropout1(tgt2)  # Add the result to the original target tensor after applying dropout.

            # Apply layer normalization after self-attention if normalization is not done before.
            if not self.normalize_before:
                tgt = self.norm1(tgt)  # Normalize after attention if specified.
            else:
                tgt = self.norm2(tgt)  # Apply second layer normalization if normalization is done before multi-head attention.
            
            # Perform multi-head attention between target and memory.
            tgt2, attn_maps = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                                key=self.with_pos_embed(memory, pos),
                                                value=memory, attn_mask=memory_mask,
                                                key_padding_mask=memory_key_padding_mask)
            
            # Add the multi-head attention output to the target tensor after applying dropout.
            tgt = tgt + self.dropout2(tgt2)

            # Apply second layer normalization again if normalization is not done before the final processing.
            if not self.normalize_before:
                tgt = self.norm2(tgt) 
            else:
                tgt = self.norm3(tgt) 

            # Apply a linear transformation followed by a dropout and then another linear transformation.
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

            # Apply second layer normalization again if normalization is not done before the final processing.
            tgt = tgt + self.dropout3(tgt2)

            if not self.normalize_before:
                tgt = self.norm3(tgt)

            return tgt, attn_maps

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, debug=False, with_state_refine=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.position_predictor = None
        self.obj_classifier = None
        self.with_state_refine = with_state_refine

        self.debug = debug

    def forward(self,
                object_queries,
                encoder_embeddings,
                encoder_embeddings_padding_mask: Optional[Tensor] = None,
                encoder_embeddings_positional_encoding: Optional[Tensor] = None,
                object_queries_positional_encoding: Optional[Tensor] = None,
                reference_points: Optional[Tensor] = None):
        """
        Computes forward propagation through the decoder, taking as input object queries and the embeddings computed by
        the decoder.

        @param object_queries: Object queries, learned vectors used as input to the decoder.
        @param encoder_embeddings: Embeddings computed in the encoder.
        @param encoder_embeddings_padding_mask: Mask signaling which embeddings correspond to pad measurements, added
            to ensure homogeneity in the measurement vector.
        @param encoder_embeddings_positional_encoding: Positional encoding added to the encoder embeddings.
        @param object_queries_positional_encoding: Positional encoding added to the object queries.
        @param reference_points: Starting points used by iterative refinement.
        @return:
            intermediate_state_predictions_normalized: State predictions computed by each of the decoder layers,
                normalized to be between 0 and 1 (or 0.25 - 0.75?)
            intermediate_logits: Logits of the existence probabilities computed by each of the decoder layers.
            debug_dict: Dictionary possibly containing useful debug information, if self.debug is True.
        """

        intermediate_state_predictions_normalized = []
        intermediate_uncertainties = []
        intermediate_logits = []
        debug_dict = {'intermediate_attention': []}

        # Go through each layer of the decoder, producing estimates and performing iterative refinement
        for layer_idx, layer in enumerate(self.layers):

            # Compute embeddings at the current decoder layer
            object_queries, attn_maps = layer(object_queries, encoder_embeddings,
                                              memory_key_padding_mask=encoder_embeddings_padding_mask,
                                              pos=encoder_embeddings_positional_encoding,
                                              query_pos=object_queries_positional_encoding)

            # Compute probability of existence using the embeddings
            predicted_logits = self.obj_classifier[layer_idx](self.norm(object_queries).permute(1, 0, 2))

            # Compute deltas using the embeddings, sum them in pre-sigmoid space
            deltas = self.pos_vel_predictor[layer_idx](self.norm(object_queries))
            normalized_predicted_state_at_current_layer = (deltas + inverse_sigmoid(reference_points)).sigmoid()

            # Compute uncertainty using the embeddings
            uncertainties = self.uncertainty_predictor[layer_idx](self.norm(object_queries))

            # If using state refinement, update the reference point using the new prediction, but detached, so as to
            # break gradient flow from here
            if self.with_state_refine:
                reference_points = normalized_predicted_state_at_current_layer.detach()

            # Save intermediate results for auxiliary loss
            intermediate_state_predictions_normalized.append(normalized_predicted_state_at_current_layer.permute(1, 0, 2))
            intermediate_uncertainties.append(uncertainties.permute(1, 0, 2))
            intermediate_logits.append(predicted_logits)

            # Save debug information if needed
            if self.debug:
                debug_dict['intermediate_attention'].append(attn_maps)


        intermediate_state_predictions_normalized = torch.stack(intermediate_state_predictions_normalized)
        intermediate_uncertainties = torch.stack(intermediate_uncertainties)
        intermediate_logits = torch.stack(intermediate_logits)
        return intermediate_state_predictions_normalized, intermediate_uncertainties, intermediate_logits, debug_dict


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

