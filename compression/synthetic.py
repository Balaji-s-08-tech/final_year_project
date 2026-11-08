"""Small random CPU fixture factory. These are not research model weights."""
import torch


def fixture():
    from VLM.quantization.awq_model_loader import _enable_project_imports
    _enable_project_imports()
    from LLMPruner.models.hf_phi2.modeling_phi import PhiForCausalLM
    from transformers import PhiConfig, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    torch.manual_seed(42)
    config=PhiConfig(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
                     max_position_embeddings=128,pad_token_id=0,bos_token_id=1,eos_token_id=2)
    model=PhiForCausalLM(config).eval()
    engine=Tokenizer(WordLevel({'[PAD]':0,'[BOS]':1,'[EOS]':2,'[UNK]':3,'calibration':4,'alpha':5,
                               'beta':6,'evaluation':7,'gamma':8,'answer':9},unk_token='[UNK]'))
    engine.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=engine,pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
    samples=[tokenizer(text,return_tensors='pt')['input_ids'] for text in ['calibration alpha','calibration beta']]
    records=[{'id':'eval-one','question':'evaluation gamma','answers':['answer']}]
    protocol={'dataset_identifier':'SYNTHETIC_FIXTURE','dataset_revision':'v1','split':'heldout_fixture','task':'text',
              'prompt_template':'{question}','model_identifier':'RANDOM_TINY_PHI','model_revision':'seed-42-v1'}
    return model,tokenizer,samples,records,protocol
