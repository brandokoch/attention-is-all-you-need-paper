import torch
import torch.nn as nn
from architectures.transformer_encoder_decoder import TransformerEncoderDecoder

class MachineTranslationTransformer(nn.Module):
    def __init__(self, d_model,n_blocks,src_vocab_size,trg_vocab_size,n_heads,d_ff, dropout_proba):
        super(MachineTranslationTransformer, self).__init__()

        self.transformer_encoder_decoder=TransformerEncoderDecoder(
            d_model,
            n_blocks,
            src_vocab_size,
            trg_vocab_size,
            n_heads,
            d_ff,
            dropout_proba
        )

    def _get_pad_mask(self, token_ids, pad_idx=0):
        pad_mask= (token_ids != pad_idx).unsqueeze(-2)
        return pad_mask.unsqueeze(1) # (batch_size, 1, 1, src_seq_len)

    def _get_lookahead_mask(self, token_ids):
        sz_b, len_s = token_ids.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=token_ids.device), diagonal=1)).bool()
        return subsequent_mask.unsqueeze(1) # (batch_size, 1, trg_seq_len, trg_seq_len)

    def forward(self, src_token_ids, trg_token_ids):

        # Since trg_token_ids contains both [BOS] and [SOS] tokens
        # we need to remove the [EOS] token when using it as input to the decoder.
        # Similarly we remove the [BOS] token when we use it as y to calculate loss,
        # which also makes y and y_pred shapes match.

        # Removing [EOS] token
        trg_token_ids=trg_token_ids[:, :-1]

        src_mask = self._get_pad_mask(src_token_ids) # (batch_size, 1, 1, src_seq_len)
        trg_mask = self._get_pad_mask(trg_token_ids) & self._get_lookahead_mask(trg_token_ids)  # (batch_size, 1, trg_seq_len, trg_seq_len)

        return self.transformer_encoder_decoder(src_token_ids, trg_token_ids, src_mask, trg_mask)

    def preprocess(self, sentence, tokenizer):
        device = next(self.parameters()).device

        src_token_ids=tokenizer.encode(sentence).ids
        src_token_ids=torch.tensor(src_token_ids, dtype=torch.long).to(device)
        src_token_ids=src_token_ids.unsqueeze(0) # To batch format

        return src_token_ids

    def translate(self, sentence, tokenizer, max_tokens=100, skip_special_tokens=False):

        # Infer the device of the model
        device = next(self.parameters()).device

        # Get tokenizer special tokens.
        eos_id=tokenizer.token_to_id('[EOS]')
        bos_id=tokenizer.token_to_id('[BOS]')

        # Tokenize sentence. 
        src_token_ids=self.preprocess(sentence, tokenizer)

        # Initialize target sequence with SOS token.
        trg_token_ids=torch.LongTensor([bos_id]).unsqueeze(0).to(device) # (1, 1)

        # Obtain src mask 
        src_mask=self._get_pad_mask(src_token_ids) # (batch_size, src_seq_len)

        # with torch.no_grad():
        encoder_output=self.transformer_encoder_decoder.encode(src_token_ids, src_mask) # (batch_size, src_seq_len, d_model)

        while True:

            # Obtain decoder output.
            trg_mask=self._get_lookahead_mask(trg_token_ids)  # Can also be set to None but for my config I found this works better.
            decoder_output=self.transformer_encoder_decoder.decode(trg_token_ids, encoder_output, src_mask, trg_mask)

            # Identify token with highest probability.
            softmax_output=nn.functional.log_softmax(decoder_output, dim=-1) # (batch_size, trg_seq_len, trg_vocab_size)
            softmax_output_last=softmax_output[:, -1, :] # (batch_size, trg_vocab_size)
            _, token_id=softmax_output_last.max(dim=-1) # (batch_size, trg_seq_len)

            # Check if token is EOS or we reached the maximum number of tokens.
            if token_id.item() == eos_id or trg_token_ids.size(1) == max_tokens:
                trg_token_ids=torch.cat([trg_token_ids, token_id.unsqueeze(0)], dim=-1) # (batch_size, trg_seq_len+1)
                break

            # Add token to target sequence.
            trg_token_ids=torch.cat([trg_token_ids, token_id.unsqueeze(0)], dim=-1) # (batch_size, trg_seq_len+1)

        # Detokenize sentence.
        decoded_output=tokenizer.decode(trg_token_ids.squeeze(0).detach().cpu().numpy(), skip_special_tokens=skip_special_tokens)

        return decoded_output

                    
        

