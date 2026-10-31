"""Local ONNX sentence encoder (CLS pooling, unit length), e.g. BAAI/bge-small-en-v1.5. CPU only, no torch."""
import glob
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class OnnxEncoder:
    def __init__(self, model_dir, max_tokens=512, batch=64):
        self.tok = Tokenizer.from_file(f'{model_dir}/tokenizer.json')
        self.tok.enable_truncation(max_tokens)
        self.tok.enable_padding()
        self.sess = ort.InferenceSession(f'{model_dir}/onnx/model.onnx', providers=['CPUExecutionProvider'])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.batch = batch

    def __call__(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            enc = self.tok.encode_batch(texts[i:i + self.batch])
            ids = np.array([e.ids for e in enc])
            feed = {'input_ids': ids, 'attention_mask': np.array([e.attention_mask for e in enc])}
            if 'token_type_ids' in self.inputs:
                feed['token_type_ids'] = np.zeros_like(ids)
            v = self.sess.run(None, feed)[0][:, 0]
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.vstack(out)


def hf_snapshot(cache_dir, repo):
    found = glob.glob(f"{cache_dir}/models--{repo.replace('/', '--')}/snapshots/*")
    if not found:
        raise FileNotFoundError(f'model {repo} not found in {cache_dir}; download it with '
                                f'`hf download {repo} --cache-dir {cache_dir}` (from huggingface_hub)')
    return found[0]
