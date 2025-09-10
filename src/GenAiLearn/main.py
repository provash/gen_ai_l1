#from tokenizer.SimpleTokenizerV1 import SimpleTokenizerV1

import re
# from tokenizer.SimpleTokenizerV2  import SimpleTokenizerV2
from dataloader.GtpDataLoaderV1 import GtpDataLoaderV1
import importlib
import importlib.metadata
import tiktoken
from dataloader.GtpDataLoaderV4 import GtpDataLoaderV4

def getRawText():
    with open("/Users/provash25/Desktop/Provash/Learning/AI/gen_ai_l1/data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text

def createVocab():

    with open("/Users/provash25/Desktop/Provash/Learning/AI/gen_ai_l1/data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        all_words = sorted(set(preprocessed))
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        vocab_size = len(all_words)

    print(vocab_size)
    vocab = {token:integer for integer,token in enumerate(all_words)}

    return vocab


def main():
     
    vocab = createVocab()
   # tokenizer = SimpleTokenizerV1(vocab)
   # tokenizer = SimpleTokenizerV2(vocab)
    tokenizer = tiktoken.get_encoding("gpt2")

    #text1 = "Hello, do you like tea?"
    #text2 = "In the sunlit terraces of the palace."
    #text = " <|endoftext|> ".join((text1, text2))

    #print("Original Text: ", text)

    #encoded = tokenizer.encode(text)
    #print("Encoded: ", encoded)

    #decoded = tokenizer.decode(encoded)
    #print("Decoded: ", decoded)

    text3 = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
    )

    integers = tokenizer.encode(text3, allowed_special={"<|endoftext|>"},disallowed_special=())
    print("Encode 2v",integers)

    strings = tokenizer.decode(integers)
    print(strings)

def test_dataloader():
    
    gtpDataLoaderV1 = GtpDataLoaderV1()

    dataloader = gtpDataLoaderV1.create_dataloader_v1(getRawText(), 4, 4, 4, False, True, 0)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)

if __name__ == "__main__":
    #print("tiktoken version:", importlib.metadata.version("tiktoken"))      
    #main()
    # Dataloader
   # test_dataloader()
    gtpDataLoaderV4 = GtpDataLoaderV4()
    gtpDataLoaderV4.test_method()


