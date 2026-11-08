import random
import json
import numpy as np
import torch
import os
from PIL import Image

from torch.utils.data.dataset import Dataset


BUNNY_DATA_PATH = os.environ.get("BUNNY_DATA_PATH", "")
LLAVA_DATA_PATH = os.environ.get("LLAVA_DATA_PATH", "")
INTERN_META_PATH = os.environ.get("INTERN_META_PATH", "")


def load_dataset(*args, **kwargs):
    from datasets import load_dataset as load
    return load(*args, **kwargs)


def _require(var_name, var_value):
    if not var_value:
        raise RuntimeError(
            f"{var_name} is not set. Export {var_name}=/path/to/dataset before calling "
            "get_examples() with a multimodal calibration dataset."
        )
    return var_value


def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len )
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

def get_alpaca(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'tatsu-lab/alpaca', split='train'
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0).to('cuda')

def prepare_bunny_samples(model, tokenizer, dataset, n_samples, seq_len, batch=False, device=None):
    """Bounded unique sampling; return per-example pairs or a padded batch triple.

    seq_len caps text tokens before image expansion. Attention masks are based on
    actual lengths, preserving genuine EOS tokens even when EOS is also padding.
    The caller owns seeding and the model; no model is loaded by this helper.
    """
    if type(n_samples) is not int or n_samples < 1 or type(seq_len) is not int or seq_len < 1:
        raise ValueError("n_samples and seq_len must be positive integers")
    if n_samples > len(dataset):
        raise ValueError("Requested more unique samples than the dataset contains")
    parameter = next(model.parameters(), None)
    target = torch.device(device or (parameter.device if parameter is not None else "cpu"))
    dtype = parameter.dtype if parameter is not None else torch.float32
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected = []
    for index in indices:
        record = dataset[index]  # one read: stochastic preprocessing stays aligned
        ids, labels, image = record.get('input_ids'), record.get('labels'), record.get('image')
        if image is None:
            continue
        if not torch.is_tensor(ids) or not torch.is_tensor(labels) or ids.ndim != 1 or labels.shape != ids.shape:
            raise ValueError("Bunny records require aligned 1-D input_ids and labels")
        ids, labels = ids[:seq_len].clone(), labels[:seq_len].clone()
        if ids.numel() < 2 or (ids == -200).sum().item() != 1 or not labels[1:].ne(-100).any():
            continue
        if not torch.is_tensor(image) or image.ndim != 3:
            raise ValueError("Bunny image must be a CHW tensor")
        selected.append((ids, labels, image))
        if len(selected) == n_samples:
            break
    if len(selected) != n_samples:
        raise ValueError(f"Found only {len(selected)} eligible image samples; requested {n_samples}")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer requires an explicit pad token")

    def expand(records):
        ids = torch.nn.utils.rnn.pad_sequence([r[0] for r in records], batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence([r[1] for r in records], batch_first=True, padding_value=-100)
        mask = torch.arange(ids.shape[1])[None, :] < torch.tensor([r[0].numel() for r in records])[:, None]
        images = [r[2].to(device=target, dtype=dtype).unsqueeze(0) for r in records]
        images = torch.cat(images, dim=0) if all(i.shape == images[0].shape for i in images) else images
        with torch.no_grad():
            _, _, expanded_mask, _, embeds, expanded_labels = model.prepare_inputs_labels_for_multimodal(
                input_ids=ids.to(target), position_ids=None, past_key_values=None,
                labels=labels.to(target), attention_mask=mask.to(target), images=images)
        if embeds is None or expanded_labels is None or expanded_mask is None:
            raise ValueError("Multimodal preparation did not return embeddings, labels and mask")
        if embeds.shape[:2] != expanded_labels.shape or expanded_labels.shape != expanded_mask.shape:
            raise ValueError("Expanded multimodal tensor shapes are inconsistent")
        if not all((row[1:] != -100).any() for row in expanded_labels):
            raise ValueError("Image expansion removed all supervised next-token targets")
        return embeds.detach(), expanded_labels.detach(), expanded_mask.detach()

    if batch:
        return expand(selected)
    return [(embeds, labels) for embeds, labels, _ in (expand([record]) for record in selected)]


def bunny_pruning_loss(model, samples, device=None):
    """Consume the batched sampler triple without treating it as a token tensor."""
    if not isinstance(samples, (tuple, list)) or len(samples) != 3:
        raise ValueError("Bunny calibration requires embeddings, labels and attention mask")
    embeds, labels, mask = samples
    if device is not None:
        embeds, labels, mask = (tensor.to(device) for tensor in samples)
    if not all((row[1:] != -100).any() for row in labels):
        raise ValueError("No supervised next-token targets")
    loss = model(inputs_embeds=embeds, labels=labels, attention_mask=mask).loss
    if not torch.isfinite(loss).all():
        raise ValueError("Nonfinite pruning loss")
    return loss


def get_bunny(tokenizer, n_samples, seq_len, batch=False, device=None):
    from bunny.util.data_utils import LazySupervisedDataset, DataArguments
    from bunny.model.builder import load_pruned_bunny_model_all
    bunny_root = _require("BUNNY_DATA_PATH", BUNNY_DATA_PATH)
    if type(n_samples) is not int or n_samples < 1 or type(seq_len) is not int or seq_len < 1:
        raise ValueError("n_samples and seq_len must be positive integers")
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, processor, _ = load_pruned_bunny_model_all("BAAI/Bunny-v1_0-3B", device=target)
    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    args = DataArguments(data_path=os.path.join(bunny_root, "bunny_695k.json"),
                         image_folder=os.path.join(bunny_root, "images"), lazy_preprocess=True, image_aspect_ratio="pad")
    args.image_processor = processor
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=args.data_path, data_args=args)
    return prepare_bunny_samples(model, tokenizer, dataset, n_samples, seq_len, batch, target)

def get_llava(tokenizer, n_samples, seq_len, batch):
    from llava.train.train_pruned import LazySupervisedDataset, DataArguments
    from llava.model.builder import load_pruned_llava_model_all
    tokenizer,model,image_processor,_ = load_pruned_llava_model_all("liuhaotian/llava-v1.5-7b",torch_dtype=torch.float16)
    tokenizer.pad_token = tokenizer.unk_token

    data_args = DataArguments()
    data_args.lazy_preprocess = True
    llava_root = _require("LLAVA_DATA_PATH", LLAVA_DATA_PATH)
    data_args.data_path = os.path.join(llava_root, "llava_v1_5_mix665k.json")
    data_args.image_folder = llava_root
    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    data_args.image_aspect_ratio = "pad"
    data_args.mm_use_im_start_end  = model.config.mm_use_im_start_end 
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    traindata = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    if not batch:
        embeds = []
        labels = []
        history = []
        for _ in range(n_samples):
            sample = {}
            while True:
                i = random.randint(0, len(traindata) - 1)
                sample['input_ids'] = traindata[i]['input_ids']
                if sample['input_ids'].shape[0] >= seq_len and i not in history and traindata[i]['image'] is not None:
                    history.append(i)
                    break
            sample['input_ids'] = sample['input_ids'].unsqueeze(0).to('cuda')
            sample['images'] = traindata[i]['image'].unsqueeze(0).half().to('cuda')
            sample['labels']  = traindata[i]['labels']
                
            _, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels = model.to("cuda").prepare_inputs_labels_for_multimodal(position_ids=None,past_key_values=None, attention_mask=None,**sample)
            labels.append(new_labels)
            embeds.append(new_input_embeds)
            labels = new_labels
            del model
            del tokenizer
            del image_processor
            torch.cuda.empty_cache()
            return embeds, labels
    else:
        input_ids = []
        images = []
        labels = []
        history = []
        print("sampling calibration data")
        for _ in range(n_samples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                input_id = traindata[i]['input_ids']
                if input_id.shape[0] <= seq_len and i not in history and traindata[i]['image'] is not None:
                    history.append(i)
                    break
            input_ids.append(traindata[i]['input_ids'])
            images.append(traindata[i]['image'])
            labels.append(traindata[i]['labels'])
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == tokenizer.eos_token_id] = -300
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
                    labels,
                    batch_first=True,
                    padding_value=-100)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        input_ids = input_ids[:tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids.to("cuda"),
            labels=labels.to("cuda"),
            attention_mask=attention_mask
        )
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images).half().to("cuda")
        else:
            batch['images'] = images.half().to("cuda")
        _, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels= model.to("cuda").prepare_inputs_labels_for_multimodal(position_ids=None,past_key_values=None,**batch) 
        embeds = new_input_embeds.detach()
        labels = new_labels
        del model
        del tokenizer
        del image_processor
        torch.cuda.empty_cache()
        return embeds, labels, attention_mask
    
def get_scienceqa(tokenizer, n_samples, seq_len):
    
    def sqa_doc_to_text(doc):
        context, question, choices = doc["hint"], doc["question"], doc["choices"]
        len_choices = len(choices)
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        if context:
            context = f"Context: {context}\n"
        post_prompt = "\nAnswer with the option's letter from the given choices directly."
        label = options[doc["answer"]]
        return f"{context}{question}\n{choices_str}{post_prompt}{label}."

    dataset = load_dataset("lmms-lab/ScienceQA", 'ScienceQA-FULL')['validation'][:100]
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(dataset) - 1)
            if dataset[i]["image"] is None:
                text = sqa_doc_to_text(dataset[i])
                tokenized_sample = tokenizer(text, return_tensors='pt')
            else:
                continue
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        tokenized_samples.append(tokenized_sample.input_ids[:, :-seq_len])
    return torch.cat(tokenized_samples, dim=0 )
    

def get_examples(dataset, tokenizer, n_samples, batch=False, seq_len = 128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == "alpaca":
        return get_alpaca(tokenizer, n_samples, seq_len)
    elif dataset == "scienceqa_txt":
        return get_scienceqa(tokenizer, n_samples, seq_len)
    elif dataset == "bunny":
        return get_bunny(tokenizer, n_samples, seq_len, batch=batch)
    elif dataset == "llava":
        return get_llava(tokenizer, n_samples, seq_len, batch=batch)
    else:
        raise NotImplementedError
