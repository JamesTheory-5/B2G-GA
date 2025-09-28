# B2G-GA

got it — here’s a minimal, end-to-end **BERT → BridgeHead → GPT-2** setup where the bridge is a tiny, differentiable GA-ish “head” (noise-gated + self-crossover) that’s trained jointly to reduce loss.

```python
# pip install transformers accelerate datasets
import torch, torch.nn as nn
from transformers import (
    BertModel, BertTokenizerFast,
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    Trainer, TrainingArguments
)

# ---------- BridgeHead: tiny GA-ish layer (no population) ----------
class BridgeHead(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.proj   = nn.Linear(enc_dim, dec_dim)      # map BERT → GPT-2 emb
        self.ln     = nn.LayerNorm(dec_dim)
        self.alpha  = nn.Parameter(torch.tensor(0.5))  # soft "crossover" gate
        self.sigma  = nn.Parameter(torch.tensor(0.05)) # learnable noise scale

    def forward(self, enc_h):  # [B, L, enc_dim]
        z = self.ln(self.proj(enc_h))                  # [B, L, dec_dim]
        # self-crossover: split last dim & blend
        z1, z2 = torch.chunk(z, 2, dim=-1)
        cross  = self.alpha * z1 + (1 - self.alpha) * z2
        # mutation-like noise (differentiable scale)
        noise  = torch.randn_like(cross) * self.sigma
        out    = torch.cat([z1, cross + noise], dim=-1)  # [B, L, dec_dim]
        return out

# ---------- Full model: BERT encoder → BridgeHead → GPT-2 decoder ----------
class Bert2Gpt2Bridge(nn.Module):
    def __init__(self, bert_name="bert-base-uncased", gpt2_name="gpt2"):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_name)

        # enable cross-attn in GPT-2 so it can attend to encoder states
        gcfg = GPT2Config.from_pretrained(gpt2_name)
        gcfg.add_cross_attention = True
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt2_name, config=gcfg)

        enc_dim = self.encoder.config.hidden_size
        dec_dim = self.decoder.config.n_embd
        assert dec_dim % 2 == 0, "bridge splits embed dim; use even n_embd"
        self.bridge = BridgeHead(enc_dim, dec_dim)

    def forward(self, enc_input_ids, enc_attn_mask, dec_input_ids, labels=None):
        enc = self.encoder(input_ids=enc_input_ids, attention_mask=enc_attn_mask)
        bridged = self.bridge(enc.last_hidden_state)  # [B, L_enc, dec_dim]
        out = self.decoder(
            input_ids=dec_input_ids,
            labels=labels,
            encoder_hidden_states=bridged,
            encoder_attention_mask=enc_attn_mask
        )
        return out  # out.loss, out.logits

# ---------- minimal training glue ----------
# tokenizers
btok  = BertTokenizerFast.from_pretrained("bert-base-uncased")
gtok  = GPT2TokenizerFast.from_pretrained("gpt2")
gtok.pad_token = gtok.eos_token  # make GPT-2 pad-safe

# tiny dummy dataset example (replace with real pairs)
from datasets import Dataset
pairs = [{"src":"what is the capital of france?", "tgt":"Paris."},
         {"src":"2+2=?", "tgt":"4."}]
def preprocess(ex):
    enc = btok(ex["src"], max_length=128, truncation=True, padding="max_length")
    dec = gtok(ex["tgt"], max_length=64,  truncation=True, padding="max_length")
    # labels = decoder input shifted (Trainer handles shift internally for GPT-2)
    ex.update({
        "enc_input_ids": enc["input_ids"],
        "enc_attn_mask": enc["attention_mask"],
        "dec_input_ids": dec["input_ids"],
        "labels":        dec["input_ids"],
    })
    return ex
ds = Dataset.from_list(pairs).map(preprocess).with_format("torch")

model = Bert2Gpt2Bridge()

class Collator:
    def __call__(self, batch):
        return {
            "enc_input_ids": torch.stack([b["enc_input_ids"] for b in batch]),
            "enc_attn_mask": torch.stack([b["enc_attn_mask"] for b in batch]),
            "dec_input_ids": torch.stack([b["dec_input_ids"] for b in batch]),
            "labels":        torch.stack([b["labels"] for b in batch]),
        }
def hf_forward(batch):
    return model(
        enc_input_ids=batch["enc_input_ids"],
        enc_attn_mask=batch["enc_attn_mask"],
        dec_input_ids=batch["dec_input_ids"],
        labels=batch["labels"],
    )

# wrap for HF Trainer
from transformers import Trainer
class Wrapped(nn.Module):
    def __init__(self, core): super().__init__(); self.core=core
    def forward(self, **batch): return hf_forward(batch)

training_args = TrainingArguments(
    output_dir="bert2gpt-bridge",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="no",
    report_to=[]
)
trainer = Trainer(
    model=Wrapped(model),
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds,
    data_collator=Collator(),
    tokenizer=gtok,
)
# trainer.train()

# ---------- inference ----------
@torch.no_grad()
def generate(text, max_new_tokens=32):
    enc = btok(text, return_tensors="pt")
    enc_h = model.encoder(**enc).last_hidden_state
    bridged = model.bridge(enc_h)
    # prime decoder with BOS (eos) token
    dec_inp = torch.tensor([[gtok.eos_token_id]])
    return gtok.decode(
        model.decoder.generate(
            input_ids=dec_inp,
            max_new_tokens=max_new_tokens,
            encoder_hidden_states=bridged,
            encoder_attention_mask=enc["attention_mask"],
            pad_token_id=gtok.eos_token_id
        )[0], skip_special_tokens=True
    )

# print(generate("what is the capital of france?"))
```

### notes

* The **BridgeHead** is the “clever” bit: a learnable, loss-driven perturb-and-blend on encoder states (self-crossover + mutation) that stays **single-path** (no population/multi-trials).
* Train end-to-end; gradients tune BERT, GPT-2, **and** the bridge (α, σ) to reduce loss/perplexity.
* For stronger effect, you can:

  * add a small *gated residual* (`out = z + gate*(cross+noise - z)`),
  * schedule or condition `σ` on layer/step,
  * or make α, σ token-wise via tiny MLPs.
