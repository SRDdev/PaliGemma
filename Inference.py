"""Inference File for PaliGemma"""

from PIL import Image
import torch
import fire

from modules.multimodal import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from modules.text_encoder import KVCache
from modules.multimodal import load_hf_model
from modules.utils import load_config
class PaliGemmaInference:
    """
    Class for Inferencing PaliGemma
    """
    def __init__(self, model_path, only_cpu=False):
        self.device = self._get_device(only_cpu)
        print("Device in use: ", self.device)
        print("-"*100)
        print(f"Loading model")
        self.model, self.tokenizer = self._load_model(model_path)
        self.processor = PaliGemmaProcessor(
            self.tokenizer,
            self.model.config.vision_config.num_image_tokens,
            self.model.config.vision_config.image_size,
        )

    def _get_device(self, only_cpu):
        if only_cpu:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self, model_path):
        model, tokenizer = load_hf_model(model_path, self.device)
        model = model.to(self.device).eval()
        return model, tokenizer

    def _move_inputs_to_device(self, model_inputs):
        return {k: v.to(self.device) for k, v in model_inputs.items()}
    
    def _get_model_inputs(self, prompt, image_file_path):
        image = Image.open(image_file_path)
        images = [image]
        prompts = [prompt]
        model_inputs = self.processor(text=prompts, images=images)
        return self._move_inputs_to_device(model_inputs)

    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        return torch.gather(probs_idx, -1, next_token)

    def test_inference(self,prompt,image_file_path,max_tokens_to_generate=100,temperature=0.8,top_p=0.9,do_sample=False):
        model_inputs = self._get_model_inputs(prompt, image_file_path)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]

        kv_cache = KVCache()
        stop_token = self.processor.tokenizer.eos_token_id
        generated_tokens = []

        for _ in range(max_tokens_to_generate):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :]
            
            if do_sample:
                next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = self._sample_top_p(next_token_logits, top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            assert next_token.size() == (1, 1)
            next_token = next_token.squeeze(0)
            generated_tokens.append(next_token)

            if next_token.item() == stop_token:
                break

            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
            )

        generated_tokens = torch.cat(generated_tokens, dim=-1)
        decoded = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(prompt + decoded)

#--------------------------------------------------------------------#
def main(model_path: str,prompt: str,image_file_path: str,max_tokens_to_generate: int = 100,
    temperature: float = 0.8,top_p: float = 0.9,do_sample: bool = False,only_cpu: bool = False):
    """
    """
    inference = PaliGemmaInference(model_path, only_cpu)
    inference.test_inference(prompt,image_file_path,max_tokens_to_generate,temperature,top_p,do_sample)

#--------------------------------------------------------------------#
if __name__ == "__main__":
    config = load_config()['PaliGemma']
    model_path = config['model_path']
    prompt = config['prompt']
    image_file_path = config['image_path']
    max_tokens_to_generate = config['max_tokens_to_generate']
    temp = config['temp']
    top_p = config['top_p']
    do_sample = config['do_sample']
    only_cpu = config['only_cpu']
    main(model_path,prompt,image_file_path,max_tokens_to_generate,temp,top_p,do_sample,only_cpu)
    